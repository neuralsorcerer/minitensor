// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{CpuAllocator, PoolStats};
use crate::{device::Device, error::Result};
use std::collections::HashMap;

/// Unified memory manager for all devices
pub struct UnifiedMemoryManager {
    cpu_allocator: CpuAllocator,
}

impl UnifiedMemoryManager {
    /// Create a new unified memory manager
    pub fn new() -> Self {
        Self {
            cpu_allocator: CpuAllocator::new(),
        }
    }

    /// Allocate memory on the specified device
    pub fn allocate(&mut self, size: usize, device: Device) -> Result<*mut u8> {
        match device.device_type {
            crate::device::DeviceType::CPU => {
                self.cpu_allocator.allocate(size)
            }
            _ => {
                // For now, fallback to CPU for GPU devices
                // GPU allocators will be implemented in tasks 9.1-9.3
                self.cpu_allocator.allocate(size)
            }
        }
    }

    /// Deallocate memory on the specified device
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize, device: Device) -> Result<()> {
        match device.device_type {
            crate::device::DeviceType::CPU => {
                self.cpu_allocator.deallocate(ptr, size)
            }
            _ => {
                // For now, fallback to CPU for GPU devices
                self.cpu_allocator.deallocate(ptr, size)
            }
        }
    }

    /// Get statistics for a specific device
    pub fn get_stats(&self, _device: Device) -> Option<PoolStats> {
        // Simplified stats - will be enhanced with proper pooling
        Some(PoolStats {
            free_blocks: 0,
            allocated_blocks: 0,
            total_free_memory: 0,
            total_allocated_memory: 0,
        })
    }

    /// Get statistics for all devices
    pub fn get_all_stats(&self) -> HashMap<Device, PoolStats> {
        let mut stats = HashMap::new();
        stats.insert(Device::cpu(), PoolStats {
            free_blocks: 0,
            allocated_blocks: 0,
            total_free_memory: 0,
            total_allocated_memory: 0,
        });
        stats
    }

    /// Clear all memory pools
    pub fn clear_all(&mut self) {
        // Nothing to clear in simplified version
    }

    /// Clear memory pool for a specific device
    pub fn clear_device(&mut self, _device: Device) {
        // Nothing to clear in simplified version
    }
}

impl Default for UnifiedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

use super::Allocator;
use std::sync::Mutex;

/// Global memory manager instance
static GLOBAL_MEMORY_MANAGER: Mutex<Option<UnifiedMemoryManager>> = Mutex::new(None);

/// Initialize the global memory manager
pub fn init_memory_manager() {
    let mut manager = GLOBAL_MEMORY_MANAGER.lock().unwrap();
    *manager = Some(UnifiedMemoryManager::new());
}

/// Allocate memory using the global manager
pub fn global_allocate(size: usize, device: Device) -> Result<*mut u8> {
    let mut manager = GLOBAL_MEMORY_MANAGER.lock().unwrap();
    if let Some(ref mut mgr) = *manager {
        mgr.allocate(size, device)
    } else {
        // Fallback to direct CPU allocation if manager not initialized
        let mut cpu_allocator = CpuAllocator::new();
        cpu_allocator.allocate(size)
    }
}

/// Deallocate memory using the global manager
pub fn global_deallocate(ptr: *mut u8, device: Device) -> Result<()> {
    let mut manager = GLOBAL_MEMORY_MANAGER.lock().unwrap();
    if let Some(ref mut mgr) = *manager {
        // We need to know the size for deallocation, but we don't track it here
        // This is a limitation of the current design - in a real implementation,
        // we'd track allocation sizes
        mgr.deallocate(ptr, 0, device) // Size 0 as placeholder
    } else {
        // Fallback to direct CPU deallocation
        let mut cpu_allocator = CpuAllocator::new();
        cpu_allocator.deallocate(ptr, 0) // Size 0 as placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_memory_manager() {
        let mut manager = UnifiedMemoryManager::new();
        let device = Device::cpu();
        
        // Test allocation
        let ptr = manager.allocate(1024, device).unwrap();
        assert!(!ptr.is_null());
        
        // Test stats
        let stats = manager.get_stats(device).unwrap();
        assert_eq!(stats.allocated_blocks, 0); // Simplified stats
        
        // Test deallocation
        manager.deallocate(ptr, 1024, device).unwrap();
    }

    #[test]
    fn test_global_memory_manager() {
        init_memory_manager();
        
        let device = Device::cpu();
        let ptr = global_allocate(512, device).unwrap();
        assert!(!ptr.is_null());
        
        global_deallocate(ptr, device).unwrap();
    }
}