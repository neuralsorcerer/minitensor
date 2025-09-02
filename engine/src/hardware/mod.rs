// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
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

        // For larger workloads, prefer GPU if available
        for gpu in &self.gpu_devices {
            if gpu.is_available {
                return match gpu.device_type {
                    DeviceType::Cuda => Device::cuda(Some(gpu.device_id)),
                    DeviceType::Metal => Device::metal(),
                    DeviceType::OpenCL => Device::opencl(Some(gpu.device_id)),
                    DeviceType::Cpu => Device::cpu(),
                };
            }
        }

        Device::cpu()
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
        let mut devices = vec![Device::cpu()];

        for gpu in gpu_devices {
            if gpu.is_available {
                let device = match gpu.device_type {
                    DeviceType::Cuda => Device::cuda(Some(gpu.device_id)),
                    DeviceType::Metal => Device::metal(),
                    DeviceType::OpenCL => Device::opencl(Some(gpu.device_id)),
                    DeviceType::Cpu => continue, // Already added CPU
                };
                devices.push(device);
            }
        }

        devices
    }
}
