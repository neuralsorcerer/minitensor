// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{Allocator, CpuAllocator};
use crate::{device::Device, error::Result};
use std::array::from_fn;
use std::ptr::NonNull;

const POOL_BUCKETS: usize = usize::BITS as usize;

/// Statistics about memory pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub total_free_memory: usize,
    pub total_allocated_memory: usize,
}

/// Memory pool that groups allocations by powers of two and reuses freed blocks.
/// The pool keeps per-size free lists indexed by the exponent of the size.
pub struct MemoryPool {
    device: Device,
    base_allocator: Box<dyn Allocator>,
    free_lists: [Vec<NonNull<u8>>; POOL_BUCKETS],
    allocated_blocks: usize,
    allocated_bytes: usize,
    free_blocks: usize,
    free_bytes: usize,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Create a new memory pool for the given device
    #[inline]
    pub fn new(device: Device) -> Self {
        let allocator: Box<dyn Allocator> = match device.device_type() {
            crate::device::DeviceType::Cpu => Box::new(CpuAllocator::new()),
            #[cfg(feature = "cuda")]
            crate::device::DeviceType::Cuda => {
                Box::new(super::CudaAllocator::new(device.device_id()))
            }
            #[cfg(feature = "metal")]
            crate::device::DeviceType::Metal => {
                Box::new(super::MetalAllocator::new(device.device_id()))
            }
            #[cfg(feature = "opencl")]
            crate::device::DeviceType::OpenCL => {
                Box::new(super::OpenCLAllocator::new(device.device_id()))
            }
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "opencl")))]
            _ => Box::new(CpuAllocator::new()),
        };

        Self {
            device,
            base_allocator: allocator,
            free_lists: from_fn(|_| Vec::new()),
            allocated_blocks: 0,
            allocated_bytes: 0,
            free_blocks: 0,
            free_bytes: 0,
        }
    }

    #[inline]
    fn bucket_for(size: usize) -> Result<(usize, usize)> {
        if size == 0 {
            return Ok((0, 0));
        }
        let rounded = size.checked_next_power_of_two().ok_or_else(|| {
            crate::error::MinitensorError::memory_error("Allocation size too large")
        })?;
        let idx = rounded.trailing_zeros() as usize;
        Ok((rounded, idx))
    }

    /// Allocate memory from the pool.
    #[inline]
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }
        let (rounded, idx) = Self::bucket_for(size)?;
        self.allocated_blocks += 1;
        self.allocated_bytes += rounded;
        if let Some(ptr) = self.free_lists[idx].pop() {
            self.free_blocks -= 1;
            self.free_bytes -= rounded;
            Ok(ptr.as_ptr())
        } else {
            self.base_allocator.allocate(rounded)
        }
    }

    /// Return memory to the pool for future reuse.
    #[inline]
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        if ptr.is_null() || size == 0 {
            return Ok(());
        }

        let (rounded, idx) = Self::bucket_for(size)?;
        self.allocated_blocks = self.allocated_blocks.saturating_sub(1);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(rounded);
        self.free_blocks += 1;
        self.free_bytes += rounded;
        self.free_lists[idx].push(unsafe { NonNull::new_unchecked(ptr) });
        Ok(())
    }

    /// Clear all cached blocks.
    #[inline]
    pub fn clear(&mut self) {
        for (idx, list) in self.free_lists.iter_mut().enumerate() {
            let size = 1usize << idx;
            for ptr in list.drain(..) {
                let _ = self.base_allocator.deallocate(ptr.as_ptr(), size);
            }
        }
        self.free_blocks = 0;
        self.free_bytes = 0;
    }

    /// Retrieve statistics about current pool usage.
    #[inline]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            free_blocks: self.free_blocks,
            allocated_blocks: self.allocated_blocks,
            total_free_memory: self.free_bytes,
            total_allocated_memory: self.allocated_bytes,
        }
    }

    /// Get the device this pool operates on.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Simple allocator wrapper backed by a [`MemoryPool`].
pub struct PooledAllocator {
    pool: MemoryPool,
}

impl PooledAllocator {
    #[inline]
    pub fn new(device: Device) -> Self {
        Self {
            pool: MemoryPool::new(device),
        }
    }
}

unsafe impl Send for PooledAllocator {}
unsafe impl Sync for PooledAllocator {}

impl Allocator for PooledAllocator {
    #[inline(always)]
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        self.pool.allocate(size)
    }

    #[inline(always)]
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        self.pool.deallocate(ptr, size)
    }

    #[inline(always)]
    fn device(&self) -> Device {
        self.pool.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_pool_reuse() {
        let mut pool = MemoryPool::new(Device::cpu());
        let ptr1 = pool.allocate(100).unwrap();
        pool.deallocate(ptr1, 100).unwrap();
        let ptr2 = pool.allocate(80).unwrap();
        assert_eq!(ptr1, ptr2);
        pool.deallocate(ptr2, 80).unwrap();
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = MemoryPool::new(Device::cpu());
        let ptr = pool.allocate(256).unwrap();
        pool.deallocate(ptr, 256).unwrap();
        assert!(pool.stats().free_blocks > 0);
        pool.clear();
        assert_eq!(pool.stats().free_blocks, 0);
        // allocating again still works
        let new_ptr = pool.allocate(256).unwrap();
        assert!(!new_ptr.is_null());
        pool.deallocate(new_ptr, 256).unwrap();
    }

    #[test]
    fn test_zero_sized_allocation() {
        let mut pool = MemoryPool::new(Device::cpu());
        let ptr = pool.allocate(0).unwrap();
        assert!(ptr.is_null());
        pool.deallocate(ptr, 0).unwrap();
    }

    #[test]
    fn test_pooled_allocator_reuse() {
        let mut alloc = PooledAllocator::new(Device::cpu());
        let ptr1 = alloc.allocate(100).unwrap();
        alloc.deallocate(ptr1, 100).unwrap();
        // 100 and 80 both round up to 128 bytes
        let ptr2 = alloc.allocate(80).unwrap();
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_concurrent_pool_usage() {
        let pool = Arc::new(Mutex::new(MemoryPool::new(Device::cpu())));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let mut pool = pool.lock();
                        let ptr = pool.allocate(128).unwrap();
                        pool.deallocate(ptr, 128).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn test_stats_tracking() {
        let mut pool = MemoryPool::new(Device::cpu());
        let a = pool.allocate(64).unwrap();
        let b = pool.allocate(128).unwrap();
        pool.deallocate(a, 64).unwrap();
        pool.deallocate(b, 128).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.free_blocks, 2);
        assert_eq!(stats.total_free_memory, 192);
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.total_allocated_memory, 0);

        let c = pool.allocate(100).unwrap();
        assert_eq!(c, b);
        let stats2 = pool.stats();
        assert_eq!(stats2.free_blocks, 1);
        assert_eq!(stats2.total_free_memory, 64);
        pool.deallocate(c, 100).unwrap();
    }

    #[test]
    fn test_pool_large_allocation_error() {
        let mut pool = MemoryPool::new(Device::cpu());
        let res = pool.allocate(usize::MAX);
        assert!(res.is_err());
    }
}
