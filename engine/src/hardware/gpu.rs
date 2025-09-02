// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::DeviceType;

/// GPU device information and capabilities
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub device_type: DeviceType,
    pub device_id: usize,
    pub name: String,
    pub vendor: String,
    pub memory_size: usize, // bytes
    pub compute_capability: ComputeCapability,
    pub max_compute_units: usize,
    pub max_work_group_size: usize,
    pub max_clock_frequency: Option<f64>, // MHz
    pub is_available: bool,
    pub capabilities: GpuCapabilities,
}

/// GPU compute capability information
#[derive(Debug, Clone)]
pub enum ComputeCapability {
    CUDA { major: u32, minor: u32 },
    Metal { family: MetalFamily },
    OpenCL { version: String },
}

/// Metal GPU families
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalFamily {
    Mac1,
    Mac2,
    MacCatalyst1,
    MacCatalyst2,
    IOs1,
    IOs2,
    IOs3,
    IOs4,
    IOs5,
    TvOs1,
    TvOs2,
    Unknown,
}

/// GPU-specific capabilities and features
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub supports_fp16: bool,
    pub supports_fp64: bool,
    pub supports_int8: bool,
    pub supports_unified_memory: bool,
    pub supports_async_compute: bool,
    pub max_texture_size: Option<(usize, usize)>,
    pub local_memory_size: usize,
    pub constant_memory_size: usize,
    pub memory_bandwidth: Option<f64>, // GB/s
}

/// GPU type enumeration for different backends
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuType {
    CUDA,
    Metal,
    OpenCL,
}

impl GpuDevice {
    /// Detect all available GPU devices
    pub fn detect_all() -> Vec<Self> {
        let devices = Vec::new();

        // Detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            devices.extend(Self::_detect_cuda_devices());
        }

        // Detect Metal devices
        #[cfg(feature = "metal")]
        {
            devices.extend(Self::_detect_metal_devices());
        }

        // Detect OpenCL devices
        #[cfg(feature = "opencl")]
        {
            devices.extend(Self::_detect_opencl_devices());
        }

        devices
    }

    #[cfg(feature = "cuda")]
    fn _detect_cuda_devices() -> Vec<Self> {
        let mut devices = Vec::new();

        // Use cudarc to detect CUDA devices
        match cudarc::driver::CudaDevice::new(0) {
            Ok(device) => {
                let device_count = device.num_devices().unwrap_or(0);
                
                for device_id in 0..device_count {
                    if let Ok(cuda_device) = cudarc::driver::CudaDevice::new(device_id) {
                        let name = cuda_device.name().unwrap_or_else(|_| format!("CUDA Device {}", device_id));
                        let memory_info = cuda_device.memory_info().unwrap_or((0, 0));
                        let total_memory = memory_info.1;

                        // Get compute capability
                        let (major, minor) = cuda_device.compute_capability().unwrap_or((0, 0));

                        devices.push(Self {
                            device_type: DeviceType::CUDA,
                            device_id,
                            name,
                            vendor: "NVIDIA".to_string(),
                            memory_size: total_memory,
                            compute_capability: ComputeCapability::CUDA { major, minor },
                            max_compute_units: 0, // Would need additional CUDA API calls
                            max_work_group_size: 1024, // Common CUDA default
                            max_clock_frequency: None,
                            is_available: true,
                            capabilities: GpuCapabilities {
                                supports_fp16: major >= 5 || (major == 5 && minor >= 3),
                                supports_fp64: true,
                                supports_int8: major >= 6,
                                supports_unified_memory: major >= 6,
                                supports_async_compute: true,
                                max_texture_size: Some((65536, 65536)),
                                local_memory_size: 48 * 1024, // 48KB shared memory typical
                                constant_memory_size: 64 * 1024, // 64KB constant memory
                                memory_bandwidth: None, // Would need device-specific lookup
                            },
                        });
                    }
                }
            }
            Err(_) => {
                // CUDA not available or no devices
            }
        }

        devices
    }

    #[cfg(not(feature = "cuda"))]
    fn _detect_cuda_devices() -> Vec<Self> {
        Vec::new()
    }

    #[cfg(feature = "metal")]
    fn _detect_metal_devices() -> Vec<Self> {
        let mut devices = Vec::new();

        // Metal detection would go here
        // This is a placeholder implementation
        #[cfg(target_os = "macos")]
        {
            // On macOS, there's typically one Metal device
            devices.push(Self {
                device_type: DeviceType::Metal,
                device_id: 0,
                name: "Apple GPU".to_string(),
                vendor: "Apple".to_string(),
                memory_size: 8 * 1024 * 1024 * 1024, // Placeholder: 8GB
                compute_capability: ComputeCapability::Metal { family: MetalFamily::Mac2 },
                max_compute_units: 32, // Placeholder
                max_work_group_size: 1024,
                max_clock_frequency: None,
                is_available: true,
                capabilities: GpuCapabilities {
                    supports_fp16: true,
                    supports_fp64: false, // Most Apple GPUs don't support fp64
                    supports_int8: true,
                    supports_unified_memory: true,
                    supports_async_compute: true,
                    max_texture_size: Some((16384, 16384)),
                    local_memory_size: 32 * 1024, // 32KB threadgroup memory
                    constant_memory_size: 64 * 1024,
                    memory_bandwidth: Some(400.0), // Placeholder bandwidth
                },
            });
        }

        devices
    }

    #[cfg(not(feature = "metal"))]
    fn _detect_metal_devices() -> Vec<Self> {
        Vec::new()
    }

    #[cfg(feature = "opencl")]
    fn _detect_opencl_devices() -> Vec<Self> {
        let mut devices = Vec::new();

        // OpenCL detection using opencl3
        if let Ok(platforms) = opencl3::platform::get_platforms() {
            for platform in platforms {
                if let Ok(device_ids) = opencl3::device::get_device_ids(platform.id(), opencl3::device::CL_DEVICE_TYPE_GPU) {
                    for (device_id, &device_id_raw) in device_ids.iter().enumerate() {
                        let device = opencl3::device::Device::new(device_id_raw);
                        let name = device.name().unwrap_or_else(|_| format!("OpenCL Device {}", device_id));
                        let vendor = device.vendor().unwrap_or_else(|_| "Unknown".to_string());
                        let memory_size = device.global_mem_size().unwrap_or(0) as usize;
                        let max_compute_units = device.max_compute_units().unwrap_or(0) as usize;
                        let max_work_group_size = device.max_work_group_size().unwrap_or(0);
                        let version = device.opencl_c_version().unwrap_or_else(|_| "Unknown".to_string());

                        devices.push(Self {
                            device_type: DeviceType::OpenCL,
                            device_id,
                            name,
                            vendor,
                            memory_size,
                            compute_capability: ComputeCapability::OpenCL { version },
                            max_compute_units,
                            max_work_group_size,
                            max_clock_frequency: device.max_clock_frequency().ok().map(|f| f as f64),
                            is_available: true,
                            capabilities: GpuCapabilities {
                                supports_fp16: device.extensions().map(|ext| ext.contains("cl_khr_fp16")).unwrap_or(false),
                                supports_fp64: device.extensions().map(|ext| ext.contains("cl_khr_fp64")).unwrap_or(false),
                                supports_int8: true, // Most OpenCL devices support int8
                                supports_unified_memory: false, // Conservative default
                                supports_async_compute: true,
                                max_texture_size: None, // OpenCL doesn't have textures in the same way
                                local_memory_size: device.local_mem_size().unwrap_or(0) as usize,
                                constant_memory_size: device.max_constant_buffer_size().unwrap_or(0) as usize,
                                memory_bandwidth: None, // Not directly available in OpenCL
                            },
                        });
                    }
                }
            }
        }

        devices
    }

    #[cfg(not(feature = "opencl"))]
    fn _detect_opencl_devices() -> Vec<Self> {
        Vec::new()
    }

    /// Check if this GPU supports a specific precision
    pub fn supports_precision(&self, precision: Precision) -> bool {
        match precision {
            Precision::FP16 => self.capabilities.supports_fp16,
            Precision::FP32 => true, // All GPUs support FP32
            Precision::FP64 => self.capabilities.supports_fp64,
            Precision::INT8 => self.capabilities.supports_int8,
        }
    }

    /// Get estimated memory bandwidth in GB/s
    pub fn memory_bandwidth(&self) -> f64 {
        self.capabilities.memory_bandwidth.unwrap_or_else(|| {
            // Rough estimates based on device type and memory size
            match self.device_type {
                DeviceType::Cuda => {
                    // Rough estimate based on memory size
                    if self.memory_size > 32 * 1024 * 1024 * 1024 { // > 32GB
                        900.0 // High-end datacenter GPU
                    } else if self.memory_size > 16 * 1024 * 1024 * 1024 { // > 16GB
                        600.0 // High-end consumer GPU
                    } else if self.memory_size > 8 * 1024 * 1024 * 1024 { // > 8GB
                        400.0 // Mid-range GPU
                    } else {
                        200.0 // Entry-level GPU
                    }
                }
                DeviceType::Metal => 400.0, // Apple Silicon estimate
                DeviceType::OpenCL => 300.0, // Conservative estimate
                DeviceType::Cpu => 50.0, // Should not happen for GPU device
            }
        })
    }

    /// Check if this device is suitable for a given workload
    pub fn is_suitable_for_workload(&self, workload_size: usize, required_memory: usize) -> bool {
        self.is_available 
            && self.memory_size >= required_memory
            && workload_size >= 1000 // GPUs are better for larger workloads
    }
}

/// Precision types supported by GPUs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP16,
    FP32,
    FP64,
    INT8,
}

impl ComputeCapability {
    /// Check if this compute capability supports a specific feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        match self {
            ComputeCapability::CUDA { major, minor } => {
                match feature {
                    "fp16" => *major >= 5 || (*major == 5 && *minor >= 3),
                    "tensor_cores" => *major >= 7,
                    "unified_memory" => *major >= 6,
                    "cooperative_groups" => *major >= 6,
                    _ => false,
                }
            }
            ComputeCapability::Metal { family: _ } => {
                match feature {
                    "fp16" => true, // Most Metal devices support fp16
                    "unified_memory" => true, // Apple Silicon has unified memory
                    "async_compute" => true,
                    _ => false,
                }
            }
            ComputeCapability::OpenCL { version: _ } => {
                // OpenCL feature support is more complex and device-dependent
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let devices = GpuDevice::detect_all();
        // Should not panic, may return empty vec if no GPUs available
        println!("Detected {} GPU devices", devices.len());
        for device in &devices {
            println!("  {}: {} ({})", device.device_type, device.name, device.vendor);
        }
    }

    #[test]
    fn test_precision_support() {
        let device = GpuDevice {
            device_type: DeviceType::Cuda,
            device_id: 0,
            name: "Test GPU".to_string(),
            vendor: "Test".to_string(),
            memory_size: 8 * 1024 * 1024 * 1024,
            compute_capability: ComputeCapability::CUDA { major: 7, minor: 5 },
            max_compute_units: 32,
            max_work_group_size: 1024,
            max_clock_frequency: Some(1500.0),
            is_available: true,
            capabilities: GpuCapabilities {
                supports_fp16: true,
                supports_fp64: true,
                supports_int8: true,
                supports_unified_memory: true,
                supports_async_compute: true,
                max_texture_size: Some((16384, 16384)),
                local_memory_size: 48 * 1024,
                constant_memory_size: 64 * 1024,
                memory_bandwidth: Some(500.0),
            },
        };

        assert!(device.supports_precision(Precision::FP16));
        assert!(device.supports_precision(Precision::FP32));
        assert!(device.supports_precision(Precision::FP64));
        assert!(device.supports_precision(Precision::INT8));
    }

    #[test]
    fn test_compute_capability_features() {
        let cuda_cap = ComputeCapability::CUDA { major: 7, minor: 5 };
        assert!(cuda_cap.supports_feature("fp16"));
        assert!(cuda_cap.supports_feature("tensor_cores"));
        assert!(cuda_cap.supports_feature("unified_memory"));

        let metal_cap = ComputeCapability::Metal { family: MetalFamily::Mac2 };
        assert!(metal_cap.supports_feature("fp16"));
        assert!(metal_cap.supports_feature("unified_memory"));
    }
}