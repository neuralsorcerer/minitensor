// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{MemoryPool, PoolStats};
use crate::{device::Device, error::Result};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::OnceLock};

/// Unified memory manager maintaining a memory pool for each device.
pub struct UnifiedMemoryManager {
    pools: HashMap<Device, MemoryPool>,
}

impl UnifiedMemoryManager {
    /// Create a new manager with no pools initialised.
    #[inline]
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    #[inline(always)]
    fn pool_for(&mut self, device: Device) -> &mut MemoryPool {
        self.pools
            .entry(device)
            .or_insert_with(|| MemoryPool::new(device))
    }

    /// Allocate memory on the specified device.
    #[inline]
    pub fn allocate(&mut self, size: usize, device: Device) -> Result<*mut u8> {
        self.pool_for(device).allocate(size)
    }

    /// Deallocate memory on the specified device.
    #[inline]
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize, device: Device) -> Result<()> {
        self.pool_for(device).deallocate(ptr, size)
    }

    /// Get statistics for a specific device.
    #[inline]
    pub fn get_stats(&self, device: Device) -> Option<PoolStats> {
        self.pools.get(&device).map(|p| p.stats())
    }

    /// Get statistics for all devices
    #[inline]
    pub fn get_all_stats(&self) -> HashMap<Device, PoolStats> {
        let mut out = HashMap::with_capacity(self.pools.len());
        for (device, pool) in &self.pools {
            out.insert(*device, pool.stats());
        }
        out
    }

    /// Clear all memory pools
    #[inline]
    pub fn clear_all(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear();
        }
    }

    /// Clear memory pool for a specific device.
    #[inline]
    pub fn clear_device(&mut self, device: Device) {
        if let Some(pool) = self.pools.get_mut(&device) {
            pool.clear();
        }
    }
}

impl Default for UnifiedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_MEMORY_MANAGER: OnceLock<Mutex<UnifiedMemoryManager>> = OnceLock::new();

/// Ensure the global manager is initialised.
#[inline]
pub fn init_memory_manager() {
    let _ = GLOBAL_MEMORY_MANAGER.get_or_init(|| Mutex::new(UnifiedMemoryManager::new()));
}

fn global_manager() -> &'static Mutex<UnifiedMemoryManager> {
    GLOBAL_MEMORY_MANAGER.get_or_init(|| Mutex::new(UnifiedMemoryManager::new()))
}

/// Allocate memory using the global manager.
#[inline]
pub fn global_allocate(size: usize, device: Device) -> Result<*mut u8> {
    let mut mgr = global_manager().lock();
    mgr.allocate(size, device)
}

/// Deallocate memory using the global manager.
#[inline]
pub fn global_deallocate(ptr: *mut u8, size: usize, device: Device) -> Result<()> {
    let mut mgr = global_manager().lock();
    mgr.deallocate(ptr, size, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_unified_memory_manager() {
        let mut manager = UnifiedMemoryManager::new();
        let device = Device::cpu();

        let ptr = manager.allocate(1024, device).unwrap();
        assert!(!ptr.is_null());

        manager.deallocate(ptr, 1024, device).unwrap();

        // stats should track allocations
        let stats = manager.get_stats(device).unwrap();
        assert_eq!(stats.free_blocks, 1);
        manager.clear_all();
        assert_eq!(manager.get_stats(device).unwrap().free_blocks, 0);
    }

    #[test]
    fn test_global_memory_manager() {
        init_memory_manager();

        let device = Device::cpu();
        let ptr = global_allocate(512, device).unwrap();
        assert!(!ptr.is_null());

        global_deallocate(ptr, 512, device).unwrap();

        // Zero-sized allocations should be handled gracefully
        let ptr = global_allocate(0, device).unwrap();
        assert!(ptr.is_null());
        global_deallocate(ptr, 0, device).unwrap();
    }

    #[test]
    fn test_concurrent_global_allocations() {
        init_memory_manager();
        let device = Device::cpu();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(move || {
                    for _ in 0..100 {
                        let ptr = global_allocate(128, device).unwrap();
                        global_deallocate(ptr, 128, device).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn test_clear_device_and_stats() {
        let mut manager = UnifiedMemoryManager::new();
        let device = Device::cpu();
        let ptr = manager.allocate(128, device).unwrap();
        manager.deallocate(ptr, 128, device).unwrap();
        let stats_before = manager.get_stats(device).unwrap();
        assert_eq!(stats_before.free_blocks, 1);
        manager.clear_device(device);
        let stats_after = manager.get_stats(device).unwrap();
        assert_eq!(stats_after.free_blocks, 0);
    }

    #[test]
    fn test_manager_large_allocation_error() {
        let mut manager = UnifiedMemoryManager::new();
        let device = Device::cpu();
        let res = manager.allocate(usize::MAX, device);
        assert!(res.is_err());
    }

    #[test]
    fn test_get_all_stats() {
        let mut manager = UnifiedMemoryManager::new();
        let device = Device::cpu();
        let ptr = manager.allocate(64, device).unwrap();
        manager.deallocate(ptr, 64, device).unwrap();
        let all = manager.get_all_stats();
        assert_eq!(all.len(), 1);
        assert_eq!(all.get(&device).unwrap().free_blocks, 1);
    }
}
