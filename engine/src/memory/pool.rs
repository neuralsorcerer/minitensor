// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{Allocator, CpuAllocator};
use crate::{device::Device, error::Result};
use std::collections::HashMap;

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool {
    device: Device,
    free_blocks: HashMap<usize, Vec<usize>>, // Free blocks by block ID
    allocated_blocks: HashMap<usize, (usize, *mut u8)>, // Map block ID to (size, ptr)
    allocator: Box<dyn Allocator>,
    next_block_id: usize,
}

impl MemoryPool {
    /// Create a new memory pool for the given device
    pub fn new(device: Device) -> Self {
        let allocator: Box<dyn Allocator> = match device.device_type {
            crate::device::DeviceType::CPU => Box::new(CpuAllocator::new()),
            #[cfg(feature = "cuda")]
            crate::device::DeviceType::CUDA => Box::new(super::CudaAllocator::new(device.device_id)),
            #[cfg(feature = "metal")]
            crate::device::DeviceType::Metal => Box::new(super::MetalAllocator::new(device.device_id)),
            #[cfg(feature = "opencl")]
            crate::device::DeviceType::OpenCL => Box::new(super::OpenCLAllocator::new(device.device_id)),
            #[cfg(not(feature = "cuda"))]
            crate::device::DeviceType::CUDA => Box::new(CpuAllocator::new()), // Fallback to CPU
            #[cfg(not(feature = "metal"))]
            crate::device::DeviceType::Metal => Box::new(CpuAllocator::new()), // Fallback to CPU
            #[cfg(not(feature = "opencl"))]
            crate::device::DeviceType::OpenCL => Box::new(CpuAllocator::new()), // Fallback to CPU
        };

        Self {
            device,
            free_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            allocator,
            next_block_id: 0,
        }
    }

    /// Get a block from the pool or allocate a new one
    pub fn get_block(&mut self, size: usize) -> Result<MemoryBlock> {
        // Round up to nearest power of 2 for better pooling
        let rounded_size = size.next_power_of_two();
        
        // Try to get a free block of the right size
        if let Some(free_list) = self.free_blocks.get_mut(&rounded_size) {
            if let Some(block_id) = free_list.pop() {
                if let Some((block_size, ptr)) = self.allocated_blocks.get(&block_id) {
                    return Ok(MemoryBlock {
                        id: block_id,
                        ptr: *ptr,
                        size: *block_size,
                        device: self.device,
                    });
                }
            }
        }

        // No free block available, allocate a new one
        let ptr = self.allocator.allocate(rounded_size)?;
        let block_id = self.next_block_id;
        self.next_block_id += 1;
        
        self.allocated_blocks.insert(block_id, (rounded_size, ptr));
        
        Ok(MemoryBlock {
            id: block_id,
            ptr,
            size: rounded_size,
            device: self.device,
        })
    }

    /// Return a block to the pool
    pub fn return_block(&mut self, block: MemoryBlock) -> Result<()> {
        if self.allocated_blocks.contains_key(&block.id) {
            // Add to free list for reuse
            self.free_blocks.entry(block.size).or_insert_with(Vec::new).push(block.id);
            Ok(())
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Attempted to return untracked memory block"
            ))
        }
    }

    /// Clear all free blocks
    pub fn clear(&mut self) {
        // Deallocate all free blocks
        for (_size, block_ids) in self.free_blocks.drain() {
            for block_id in block_ids {
                if let Some((size, ptr)) = self.allocated_blocks.remove(&block_id) {
                    let _ = self.allocator.deallocate(ptr, size);
                }
            }
        }
        
        // Note: We don't deallocate allocated blocks as they're still in use
        // The caller is responsible for returning them first
    }

    /// Get the device this pool operates on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> PoolStats {
        let free_blocks: usize = self.free_blocks.values().map(|v| v.len()).sum();
        let allocated_blocks = self.allocated_blocks.len();
        let total_free_memory: usize = self.free_blocks.iter()
            .map(|(size, blocks)| size * blocks.len())
            .sum();
        let total_allocated_memory: usize = self.allocated_blocks.values().map(|(size, _)| *size).sum();

        PoolStats {
            free_blocks,
            allocated_blocks,
            total_free_memory,
            total_allocated_memory,
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Clean up all memory when the pool is dropped
        self.clear();
        
        // Also deallocate any remaining allocated blocks
        // This is a safety measure - ideally all blocks should be returned
        for (_block_id, (size, ptr)) in self.allocated_blocks.drain() {
            let _ = self.allocator.deallocate(ptr, size);
        }
    }
}

/// A memory block handle that's safe to send between threads
#[derive(Debug)]
pub struct MemoryBlock {
    pub id: usize,
    pub ptr: *mut u8,
    pub size: usize,
    pub device: Device,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub total_free_memory: usize,
    pub total_allocated_memory: usize,
}

/// Pooled allocator that uses a simple in-memory free list.
pub struct PooledAllocator {
    device: Device,
    free_list: Vec<(*mut u8, usize)>,
}

impl PooledAllocator {
    /// Create a new pooled allocator
    pub fn new(device: Device) -> Self {
        Self { device, free_list: Vec::new() }
    }
}

impl Allocator for PooledAllocator {
    fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        if let Some(pos) = self.free_list.iter().position(|&(_, s)| s >= size) {
            let (ptr, _) = self.free_list.remove(pos);
            Ok(ptr)
        } else {
            let mut cpu_allocator = CpuAllocator::new();
            cpu_allocator.allocate(size)
        }
    }

    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        if ptr.is_null() || size == 0 {
            return Ok(());
        }
        self.free_list.push((ptr, size));
        Ok(())
    }

    fn device(&self) -> Device {
        self.device
    }
}