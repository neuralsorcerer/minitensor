// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Backend;
use crate::{device::Device, error::Result};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// CPU backend for tensor operations
pub struct CpuBackend {
    device: Device,
    pool: Mutex<FxHashMap<usize, Vec<NonNull<u8>>>>,
}

// The memory pool uses raw pointers protected by a mutex. As long as the API
// is used correctly, it is safe to send across threads.
unsafe impl Send for CpuBackend {}
unsafe impl Sync for CpuBackend {}

impl Backend for CpuBackend {
    #[inline(always)]
    fn device(&self) -> Device {
        self.device
    }

    #[inline(always)]
    fn is_available() -> bool {
        true // CPU is always available
    }

    #[inline(always)]
    fn initialize() -> Result<Self> {
        Ok(Self {
            device: Device::cpu(),
            pool: Mutex::new(FxHashMap::default()),
        })
    }

    #[inline(always)]
    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        // Try to reuse memory from the pool first to avoid allocator calls.
        if let Some(ptr) = {
            let mut pool = self.pool.lock();
            pool.get_mut(&size_bytes).and_then(|v| v.pop())
        } {
            return Ok(ptr.as_ptr());
        }

        // Checked, not `from_size_align_unchecked`: a size within 63 bytes of
        // `isize::MAX` is not a valid layout at this alignment, and building
        // one anyway is undefined behavior rather than an error.
        let layout = Layout::from_size_align(size_bytes, 64).map_err(|_| {
            crate::error::MinitensorError::memory_error(format!(
                "Invalid memory layout for size {size_bytes}"
            ))
        })?;

        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Failed to allocate memory",
            ));
        }

        Ok(ptr)
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: *mut u8, size_bytes: usize) -> Result<()> {
        if ptr.is_null() || size_bytes == 0 {
            return Ok(());
        }

        // Return the memory to the pool for potential reuse. Actual freeing
        // happens when the backend is dropped.
        let mut pool = self.pool.lock();
        // Safety: ptr is not null and size_bytes > 0
        pool.entry(size_bytes)
            .or_default()
            .push(unsafe { NonNull::new_unchecked(ptr) });

        Ok(())
    }

    #[inline(always)]
    unsafe fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer",
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer",
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {
        let mut pool = self.pool.lock();
        for (size, mut vec) in pool.drain() {
            if size == 0 {
                continue;
            }
            // Every block in the pool came from `allocate` with this size, so
            // the layout was valid when it was made.
            let Ok(layout) = Layout::from_size_align(size, 64) else {
                continue;
            };
            for ptr in vec.drain(..) {
                unsafe { dealloc(ptr.as_ptr(), layout) };
            }
        }
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

        unsafe { backend.deallocate(ptr, 1024) }.unwrap();
    }

    #[test]
    fn test_memory_copy() {
        let backend = CpuBackend::initialize().unwrap();

        let src_data = vec![1u8, 2, 3, 4, 5];
        let ptr = backend.allocate(5).unwrap();

        unsafe { backend.copy_from_host(ptr, &src_data) }.unwrap();

        let mut dst_data = vec![0u8; 5];
        unsafe { backend.copy_to_host(&mut dst_data, ptr) }.unwrap();

        assert_eq!(src_data, dst_data);

        unsafe { backend.deallocate(ptr, 5) }.unwrap();
    }

    #[test]
    fn test_zero_allocation_and_copy() {
        let backend = CpuBackend::initialize().unwrap();

        let ptr = backend.allocate(0).unwrap();
        assert!(ptr.is_null());

        unsafe { backend.copy_from_host(ptr, &[]) }.unwrap();
        unsafe { backend.copy_to_host(&mut [], ptr) }.unwrap();

        unsafe { backend.deallocate(ptr, 0) }.unwrap();
    }

    #[test]
    fn test_null_pointer_errors() {
        let backend = CpuBackend::initialize().unwrap();

        // Non-empty copy with null destination should error
        assert!(unsafe { backend.copy_from_host(std::ptr::null_mut(), &[1u8]) }.is_err());

        // Non-empty copy with null source should error
        let mut buf = [0u8; 1];
        assert!(unsafe { backend.copy_to_host(&mut buf, std::ptr::null()) }.is_err());
    }

    #[test]
    fn test_multiple_allocations_and_copies() {
        let backend = CpuBackend::initialize().unwrap();

        let ptr1 = backend.allocate(4).unwrap();
        let ptr2 = backend.allocate(4).unwrap();

        let data1 = [1u8, 2, 3, 4];
        let data2 = [5u8, 6, 7, 8];

        unsafe { backend.copy_from_host(ptr1, &data1) }.unwrap();
        unsafe { backend.copy_from_host(ptr2, &data2) }.unwrap();

        let mut out1 = [0u8; 4];
        let mut out2 = [0u8; 4];
        unsafe { backend.copy_to_host(&mut out1, ptr1) }.unwrap();
        unsafe { backend.copy_to_host(&mut out2, ptr2) }.unwrap();

        assert_eq!(data1, out1);
        assert_eq!(data2, out2);

        unsafe { backend.deallocate(ptr1, 4) }.unwrap();
        unsafe { backend.deallocate(ptr2, 4) }.unwrap();
    }

    #[test]
    fn test_zero_length_copy_to_valid_pointer() {
        let backend = CpuBackend::initialize().unwrap();

        let ptr = backend.allocate(8).unwrap();
        unsafe { backend.copy_from_host(ptr, &[]) }.unwrap();
        let mut buf = [0u8; 0];
        unsafe { backend.copy_to_host(&mut buf, ptr) }.unwrap();
        unsafe { backend.deallocate(ptr, 8) }.unwrap();
    }

    #[test]
    fn test_deallocate_null_pointer() {
        let backend = CpuBackend::initialize().unwrap();
        unsafe { backend.deallocate(std::ptr::null_mut(), 128) }.unwrap();
    }

    #[test]
    fn test_memory_pool_reuse() {
        let backend = CpuBackend::initialize().unwrap();

        let ptr1 = backend.allocate(128).unwrap();
        unsafe { backend.deallocate(ptr1, 128) }.unwrap();
        let ptr2 = backend.allocate(128).unwrap();

        // Pointer should be reused from pool
        assert_eq!(ptr1, ptr2);

        unsafe { backend.deallocate(ptr2, 128) }.unwrap();
    }

    #[test]
    fn test_large_memory_copy() {
        let backend = CpuBackend::initialize().unwrap();
        let size = 1 << 20; // 1 MiB
        let src: Vec<u8> = (0..size as u32).map(|i| (i & 0xFF) as u8).collect();
        let ptr = backend.allocate(size).unwrap();

        unsafe { backend.copy_from_host(ptr, &src) }.unwrap();

        let mut dst = vec![0u8; size];
        unsafe { backend.copy_to_host(&mut dst, ptr) }.unwrap();

        assert_eq!(src, dst);
        unsafe { backend.deallocate(ptr, size) }.unwrap();
    }

    #[test]
    fn test_concurrent_pool_usage() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(CpuBackend::initialize().unwrap());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let b = backend.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let ptr = b.allocate(256).unwrap();
                    unsafe { b.deallocate(ptr, 256) }.unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}
