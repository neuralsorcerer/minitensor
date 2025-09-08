// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod optimizer;
pub mod profiler;

pub use cpu::{CpuFeatures, CpuInfo, SIMDSupport};
pub use gpu::{ComputeCapability, GpuCapabilities, GpuDevice, GpuType};
pub use memory::{CacheInfo, MemoryBandwidth, MemoryInfo};
pub use optimizer::{
    AllocationStrategy, DevicePlacement, ExecutionPlan, MemoryOptimizationPlan,
    ParallelizationStrategy, ResourceOptimizer, WorkloadAnalysis,
};
pub use profiler::{HardwareProfile, HardwareProfiler, SystemInfo};

use crate::device::{Device, DeviceType};

/// System-wide hardware information
#[derive(Debug, Clone)]
pub struct SystemHardware {
    pub cpu_info: CpuInfo,
    pub gpu_devices: Vec<GpuDevice>,
    pub memory_info: MemoryInfo,
    pub available_devices: Vec<Device>,
}

impl SystemHardware {
    /// Detect and profile all available hardware
    pub fn detect() -> Self {
        let profiler = HardwareProfiler::new();
        let profile = profiler.profile_system();

        Self {
            cpu_info: profile.cpu_info,
            gpu_devices: profile.gpu_devices.clone(),
            memory_info: profile.memory_info,
            available_devices: Self::enumerate_devices(&profile.gpu_devices),
        }
    }

    /// Get the best device for a given workload size
    pub fn optimal_device(&self, workload_size: usize) -> Device {
        // For small workloads, prefer CPU
        if workload_size < 1000 {
            return Device::cpu();
        }

        // For larger workloads, prefer the first available GPU
        self.gpu_devices
            .iter()
            .find(|gpu| gpu.is_available)
            .map(|gpu| match gpu.device_type {
                DeviceType::Cuda => Device::cuda(Some(gpu.device_id)),
                DeviceType::Metal => Device::metal(),
                DeviceType::OpenCL => Device::opencl(Some(gpu.device_id)),
                DeviceType::Cpu => Device::cpu(),
            })
            .unwrap_or_else(Device::cpu)
    }

    /// Get memory capacity for a specific device
    pub fn device_memory(&self, device: &Device) -> Option<usize> {
        match device.device_type() {
            DeviceType::Cpu => Some(self.memory_info.total_ram),
            _ => self
                .gpu_devices
                .iter()
                .find(|gpu| gpu.device_id == device.id())
                .map(|gpu| gpu.memory_size),
        }
    }

    fn enumerate_devices(gpu_devices: &[GpuDevice]) -> Vec<Device> {
        let mut devices = Vec::with_capacity(gpu_devices.len() + 1);
        devices.push(Device::cpu());

        devices.extend(gpu_devices.iter().filter_map(|gpu| {
            if !gpu.is_available {
                return None;
            }
            match gpu.device_type {
                DeviceType::Cuda => Some(Device::cuda(Some(gpu.device_id))),
                DeviceType::Metal => Some(Device::metal()),
                DeviceType::OpenCL => Some(Device::opencl(Some(gpu.device_id))),
                DeviceType::Cpu => None,
            }
        }));

        devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_device_selection() {
        let cpu_info = CpuInfo {
            model_name: String::new(),
            vendor: String::new(),
            cores: 4,
            threads: 8,
            base_frequency: None,
            max_frequency: None,
            features: CpuFeatures::default(),
            cache_info: Vec::new(),
        };

        let gpu = GpuDevice {
            device_type: DeviceType::Cuda,
            device_id: 0,
            name: String::new(),
            vendor: String::new(),
            memory_size: 8 * 1024 * 1024 * 1024,
            compute_capability: ComputeCapability::CUDA { major: 7, minor: 5 },
            max_compute_units: 1,
            max_work_group_size: 1024,
            max_clock_frequency: None,
            is_available: true,
            capabilities: GpuCapabilities {
                supports_fp16: true,
                supports_fp64: true,
                supports_int8: true,
                supports_unified_memory: true,
                supports_async_compute: true,
                max_texture_size: None,
                local_memory_size: 0,
                constant_memory_size: 0,
                memory_bandwidth: None,
            },
        };

        let memory_info = MemoryInfo {
            total_ram: 8 * 1024 * 1024 * 1024,
            available_ram: 8 * 1024 * 1024 * 1024,
            total_swap: 0,
            available_swap: 0,
            page_size: 4096,
            bandwidth: MemoryBandwidth {
                sequential_read: 1.0,
                sequential_write: 1.0,
                random_read: 1.0,
                random_write: 1.0,
                copy_bandwidth: 1.0,
            },
            cache_info: Vec::new(),
        };

        let system = SystemHardware {
            cpu_info,
            gpu_devices: vec![gpu],
            memory_info,
            available_devices: vec![Device::cpu(), Device::cuda(Some(0))],
        };

        assert_eq!(system.optimal_device(10), Device::cpu());
        assert_eq!(system.optimal_device(10_000), Device::cuda(Some(0)));
    }

    #[test]
    fn test_device_memory_lookup() {
        let cpu_info = CpuInfo {
            model_name: String::new(),
            vendor: String::new(),
            cores: 4,
            threads: 8,
            base_frequency: None,
            max_frequency: None,
            features: CpuFeatures::default(),
            cache_info: Vec::new(),
        };
        let gpu = GpuDevice {
            device_type: DeviceType::Cuda,
            device_id: 1,
            name: String::new(),
            vendor: String::new(),
            memory_size: 4 * 1024 * 1024 * 1024,
            compute_capability: ComputeCapability::CUDA { major: 7, minor: 5 },
            max_compute_units: 0,
            max_work_group_size: 0,
            max_clock_frequency: None,
            is_available: true,
            capabilities: GpuCapabilities {
                supports_fp16: true,
                supports_fp64: true,
                supports_int8: true,
                supports_unified_memory: true,
                supports_async_compute: true,
                max_texture_size: None,
                local_memory_size: 0,
                constant_memory_size: 0,
                memory_bandwidth: None,
            },
        };
        let memory_info = MemoryInfo {
            total_ram: 16 * 1024 * 1024 * 1024,
            available_ram: 16 * 1024 * 1024 * 1024,
            total_swap: 0,
            available_swap: 0,
            page_size: 4096,
            bandwidth: MemoryBandwidth {
                sequential_read: 1.0,
                sequential_write: 1.0,
                random_read: 1.0,
                random_write: 1.0,
                copy_bandwidth: 1.0,
            },
            cache_info: Vec::new(),
        };
        let system = SystemHardware {
            cpu_info,
            gpu_devices: vec![gpu.clone()],
            memory_info,
            available_devices: vec![Device::cpu(), Device::cuda(Some(1))],
        };
        assert_eq!(
            system.device_memory(&Device::cuda(Some(1))),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            system.device_memory(&Device::cuda(Some(2))),
            None
        );
        assert_eq!(
            system.device_memory(&Device::cpu()),
            Some(16 * 1024 * 1024 * 1024)
        );
    }
}
