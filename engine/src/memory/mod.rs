// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
};
use std::alloc::{alloc, dealloc, Layout};

/// Global memory allocator for tensors
pub fn global_allocate(size: usize, device: Device) -> Result<*mut u8> {
    if device.is_cpu() {
        // Use system allocator for CPU
        let layout = Layout::from_size_align(size, 8)
            .map_err(|_| MinitensorError::memory_error("Invalid memory layout"))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            Err(MinitensorError::memory_error(
                "Failed to allocate CPU memory",
            ))
        } else {
            Ok(ptr)
        }
    } else {
        // For GPU devices, we'll implement proper allocation in backend tasks
        // For now, fallback to CPU allocation
        global_allocate(size, Device::cpu())
    }
}

/// Global memory deallocator for tensors
pub fn global_deallocate(ptr: *mut u8, device: Device) -> Result<()> {
    if device.is_cpu() {
        // We need to know the original layout to deallocate properly
        // For now, we'll assume a simple deallocation
        // In a real implementation, we'd track allocations
        unsafe {
            // This is unsafe because we don't know the original layout
            // In practice, we'd need to track allocations with their layouts
            let layout = Layout::from_size_align(1, 8).unwrap(); // Placeholder
            dealloc(ptr, layout);
        }
        Ok(())
    } else {
        // For GPU devices, implement proper deallocation in backend tasks
        Ok(())
    }
}

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool {
    device: Device,
    allocated_blocks: Vec<(*mut u8, usize)>,
}

impl MemoryPool {
    /// Create a new memory pool for the specified device
    pub fn new(device: Device) -> Self {
        Self {
            device,
            allocated_blocks: Vec::new(),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        // Simple implementation - just use global allocator
        // In a real implementation, we'd maintain free blocks and reuse them
        let ptr = global_allocate(size, self.device)?;
        self.allocated_blocks.push((ptr, size));
        Ok(ptr)
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, ptr: *mut u8) -> Result<()> {
        // Find and remove the block
        if let Some(pos) = self.allocated_blocks.iter().position(|(p, _)| *p == ptr) {
            self.allocated_blocks.remove(pos);
            global_deallocate(ptr, self.device)
        } else {
            Err(MinitensorError::memory_error(
                "Pointer not found in memory pool",
            ))
        }
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.allocated_blocks.iter().map(|(_, size)| *size).sum()
    }

    /// Get number of allocated blocks
    pub fn num_blocks(&self) -> usize {
        self.allocated_blocks.len()
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Clean up all allocated blocks
        for (ptr, _) in &self.allocated_blocks {
            let _ = global_deallocate(*ptr, self.device);
        }
    }
}

/// Memory statistics for monitoring
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub num_allocations: usize,
    pub num_deallocations: usize,
}

impl MemoryStats {
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            peak_allocated: 0,
            num_allocations: 0,
            num_deallocations: 0,
        }
    }

    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.peak_allocated = self.peak_allocated.max(self.total_allocated);
        self.num_allocations += 1;
    }

    pub fn record_deallocation(&mut self, size: usize) {
        self.total_allocated = self.total_allocated.saturating_sub(size);
        self.num_deallocations += 1;
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocation() {
        let ptr = global_allocate(1024, Device::cpu()).unwrap();
        assert!(!ptr.is_null());

        // Write some data to verify allocation works
        unsafe {
            *ptr = 42;
            assert_eq!(*ptr, 42);
        }

        let _ = global_deallocate(ptr, Device::cpu());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(Device::cpu());

        let ptr1 = pool.allocate(512).unwrap();
        let ptr2 = pool.allocate(1024).unwrap();

        assert_eq!(pool.num_blocks(), 2);
        assert_eq!(pool.total_allocated(), 1536);

        pool.deallocate(ptr1).unwrap();
        assert_eq!(pool.num_blocks(), 1);
        assert_eq!(pool.total_allocated(), 1024);

        pool.deallocate(ptr2).unwrap();
        assert_eq!(pool.num_blocks(), 0);
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();

        stats.record_allocation(1024);
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.peak_allocated, 1024);
        assert_eq!(stats.num_allocations, 1);

        stats.record_allocation(512);
        assert_eq!(stats.total_allocated, 1536);
        assert_eq!(stats.peak_allocated, 1536);
        assert_eq!(stats.num_allocations, 2);

        stats.record_deallocation(512);
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.peak_allocated, 1536); // Peak should remain
        assert_eq!(stats.num_deallocations, 1);
    }
}
