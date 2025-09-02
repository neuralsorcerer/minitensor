// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{CpuInfo, GpuDevice, MemoryInfo};
use crate::device::DeviceType;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Complete hardware profile of the system
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_info: CpuInfo,
    pub gpu_devices: Vec<GpuDevice>,
    pub memory_info: MemoryInfo,
    pub system_info: SystemInfo,
    pub performance_characteristics: PerformanceCharacteristics,
}

/// System information and environment
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os_name: String,
    pub os_version: String,
    pub architecture: String,
    pub hostname: String,
    pub uptime: Duration,
    pub load_average: Option<(f64, f64, f64)>, // 1min, 5min, 15min
}

/// Performance characteristics and benchmarks
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    pub cpu_single_thread_score: f64,
    pub cpu_multi_thread_score: f64,
    pub memory_latency: Duration,
    pub memory_throughput: f64, // GB/s
    pub gpu_compute_scores: HashMap<usize, f64>, // device_id -> score
    pub thermal_characteristics: ThermalInfo,
}

/// Thermal information and limits
#[derive(Debug, Clone)]
pub struct ThermalInfo {
    pub cpu_temperature: Option<f64>, // Celsius
    pub gpu_temperatures: HashMap<usize, f64>, // device_id -> temperature
    pub thermal_throttling_detected: bool,
    pub cooling_capability: CoolingCapability,
}

/// System cooling capability assessment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoolingCapability {
    Excellent, // Desktop with good cooling
    Good,      // Laptop with adequate cooling
    Limited,   // Thin laptop or mobile device
    Poor,      // Thermally constrained device
    Unknown,
}

/// Hardware profiler for system analysis
pub struct HardwareProfiler {
    _benchmark_duration: Duration,
    detailed_profiling: bool,
}

impl HardwareProfiler {
    /// Create a new hardware profiler with default settings
    pub fn new() -> Self {
        Self {
            _benchmark_duration: Duration::from_millis(100), // Quick benchmarks
            detailed_profiling: false,
        }
    }

    /// Create a profiler with detailed benchmarking enabled
    pub fn with_detailed_profiling(benchmark_duration: Duration) -> Self {
        Self {
            _benchmark_duration: benchmark_duration,
            detailed_profiling: true,
        }
    }

    /// Profile the entire system hardware
    pub fn profile_system(&self) -> HardwareProfile {
        println!("Profiling system hardware...");
        
        let cpu_info = CpuInfo::detect();
        let gpu_devices = GpuDevice::detect_all();
        let memory_info = MemoryInfo::detect();
        let system_info = SystemInfo::detect();
        
        let performance_characteristics = if self.detailed_profiling {
            self.benchmark_performance(&cpu_info, &gpu_devices, &memory_info)
        } else {
            self.quick_performance_assessment(&cpu_info, &gpu_devices, &memory_info)
        };

        HardwareProfile {
            cpu_info,
            gpu_devices,
            memory_info,
            system_info,
            performance_characteristics,
        }
    }

    /// Quick performance assessment without intensive benchmarking
    fn quick_performance_assessment(
        &self,
        cpu_info: &CpuInfo,
        gpu_devices: &[GpuDevice],
        memory_info: &MemoryInfo,
    ) -> PerformanceCharacteristics {
        // Estimate performance based on hardware specs
        let cpu_single_thread_score = self.estimate_cpu_single_thread_performance(cpu_info);
        let cpu_multi_thread_score = cpu_single_thread_score * (cpu_info.cores as f64 * 0.8);
        
        let memory_latency = Duration::from_nanos(100); // Typical DDR4 latency
        let memory_throughput = memory_info.bandwidth.sequential_read;

        let mut gpu_compute_scores = HashMap::new();
        for gpu in gpu_devices {
            let score = self.estimate_gpu_performance(gpu);
            gpu_compute_scores.insert(gpu.device_id, score);
        }

        PerformanceCharacteristics {
            cpu_single_thread_score,
            cpu_multi_thread_score,
            memory_latency,
            memory_throughput,
            gpu_compute_scores,
            thermal_characteristics: ThermalInfo::detect(),
        }
    }

    /// Detailed performance benchmarking
    fn benchmark_performance(
        &self,
        cpu_info: &CpuInfo,
        gpu_devices: &[GpuDevice],
        memory_info: &MemoryInfo,
    ) -> PerformanceCharacteristics {
        println!("Running detailed performance benchmarks...");
        
        let cpu_single_thread_score = self.benchmark_cpu_single_thread();
        let cpu_multi_thread_score = self.benchmark_cpu_multi_thread(cpu_info.cores);
        let memory_latency = self.benchmark_memory_latency();
        let memory_throughput = memory_info.bandwidth.sequential_read;

        let mut gpu_compute_scores = HashMap::new();
        for gpu in gpu_devices {
            if gpu.is_available {
                let score = self.benchmark_gpu_compute(gpu);
                gpu_compute_scores.insert(gpu.device_id, score);
            }
        }

        PerformanceCharacteristics {
            cpu_single_thread_score,
            cpu_multi_thread_score,
            memory_latency,
            memory_throughput,
            gpu_compute_scores,
            thermal_characteristics: ThermalInfo::detect(),
        }
    }

    fn estimate_cpu_single_thread_performance(&self, cpu_info: &CpuInfo) -> f64 {
        // Rough estimation based on frequency and architecture
        let base_score = cpu_info.base_frequency.unwrap_or(2500.0) / 1000.0; // Normalize to GHz
        
        // Adjust for SIMD capabilities
        let simd_multiplier = match cpu_info.features.simd_support {
            super::SIMDSupport::AVX512 => 2.0,
            super::SIMDSupport::AVX2 => 1.5,
            super::SIMDSupport::AVX => 1.3,
            super::SIMDSupport::SSE4_2 => 1.1,
            _ => 1.0,
        };

        base_score * simd_multiplier
    }

    fn estimate_gpu_performance(&self, gpu: &GpuDevice) -> f64 {
        // Rough estimation based on compute units and memory
        let compute_score = gpu.max_compute_units as f64 * 10.0;
        let memory_score = (gpu.memory_size as f64 / (1024.0 * 1024.0 * 1024.0)) * 100.0; // GB
        let bandwidth_score = gpu.memory_bandwidth() / 10.0;

        (compute_score + memory_score + bandwidth_score) / 3.0
    }

    fn benchmark_cpu_single_thread(&self) -> f64 {
        let start = Instant::now();
        let iterations = 1_000_000;
        
        // Simple floating-point benchmark
        let mut result = 1.0f64;
        for i in 0..iterations {
            result = result.sqrt() + (i as f64).sin();
        }
        
        // Prevent optimization
        std::hint::black_box(result);
        
        let elapsed = start.elapsed();
        let ops_per_second = iterations as f64 / elapsed.as_secs_f64();
        ops_per_second / 1_000_000.0 // Normalize to millions of ops per second
    }

    fn benchmark_cpu_multi_thread(&self, num_cores: usize) -> f64 {
        use rayon::prelude::*;
        
        let start = Instant::now();
        let iterations_per_thread = 100_000;
        
        // Parallel floating-point benchmark
        let results: Vec<f64> = (0..num_cores)
            .into_par_iter()
            .map(|thread_id| {
                let mut result = (thread_id + 1) as f64;
                for i in 0..iterations_per_thread {
                    result = result.sqrt() + (i as f64).sin();
                }
                result
            })
            .collect();
        
        // Prevent optimization
        std::hint::black_box(results);
        
        let elapsed = start.elapsed();
        let total_ops = num_cores * iterations_per_thread;
        let ops_per_second = total_ops as f64 / elapsed.as_secs_f64();
        ops_per_second / 1_000_000.0 // Normalize to millions of ops per second
    }

    fn benchmark_memory_latency(&self) -> Duration {
        let size = 1024 * 1024; // 1MB buffer
        let buffer = vec![0u64; size / 8];
        let iterations = 1000;
        
        let start = Instant::now();
        
        // Random memory access pattern to measure latency
        let mut sum = 0u64;
        let mut index = 0;
        for _ in 0..iterations {
            sum += buffer[index];
            index = (index + 1009) % buffer.len(); // Prime number for pseudo-random access
        }
        
        // Prevent optimization
        std::hint::black_box(sum);
        
        let total_time = start.elapsed();
        total_time / iterations
    }

    fn benchmark_gpu_compute(&self, gpu: &GpuDevice) -> f64 {
        // Placeholder GPU benchmark - would need actual GPU compute kernels
        match gpu.device_type {
            DeviceType::Cuda => {
                // Would run CUDA benchmark kernel
                self.estimate_gpu_performance(gpu) * 1.2 // Assume CUDA is 20% faster
            }
            DeviceType::Metal => {
                // Would run Metal compute shader
                self.estimate_gpu_performance(gpu) * 1.1 // Assume Metal is 10% faster
            }
            DeviceType::OpenCL => {
                // Would run OpenCL kernel
                self.estimate_gpu_performance(gpu)
            }
            DeviceType::Cpu => 0.0, // Should not happen for GPU device
        }
    }

    /// Generate a performance report
    pub fn generate_report(&self, profile: &HardwareProfile) -> String {
        let mut report = String::new();
        
        report.push_str("=== Hardware Profile Report ===\n\n");
        
        // System Information
        report.push_str(&format!("System: {} {} ({})\n", 
            profile.system_info.os_name, 
            profile.system_info.os_version,
            profile.system_info.architecture
        ));
        report.push_str(&format!("Hostname: {}\n", profile.system_info.hostname));
        report.push_str(&format!("Uptime: {:.1} hours\n\n", 
            profile.system_info.uptime.as_secs_f64() / 3600.0
        ));

        // CPU Information
        report.push_str("=== CPU Information ===\n");
        report.push_str(&format!("Model: {} ({})\n", 
            profile.cpu_info.model_name, 
            profile.cpu_info.vendor
        ));
        report.push_str(&format!("Cores: {} (Threads: {})\n", 
            profile.cpu_info.cores, 
            profile.cpu_info.threads
        ));
        if let Some(freq) = profile.cpu_info.base_frequency {
            report.push_str(&format!("Base Frequency: {:.1} MHz\n", freq));
        }
        report.push_str(&format!("SIMD Support: {:?}\n", 
            profile.cpu_info.features.simd_support
        ));
        report.push_str(&format!("Single-thread Score: {:.1}\n", 
            profile.performance_characteristics.cpu_single_thread_score
        ));
        report.push_str(&format!("Multi-thread Score: {:.1}\n\n", 
            profile.performance_characteristics.cpu_multi_thread_score
        ));

        // Memory Information
        report.push_str("=== Memory Information ===\n");
        report.push_str(&format!("Total RAM: {:.1} GB\n", 
            profile.memory_info.total_ram as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        report.push_str(&format!("Available RAM: {:.1} GB\n", 
            profile.memory_info.available_ram as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        report.push_str(&format!("Memory Bandwidth: {:.1} GB/s\n", 
            profile.performance_characteristics.memory_throughput
        ));
        report.push_str(&format!("Memory Latency: {:.1} ns\n\n", 
            profile.performance_characteristics.memory_latency.as_nanos()
        ));

        // GPU Information
        if !profile.gpu_devices.is_empty() {
            report.push_str("=== GPU Information ===\n");
            for gpu in &profile.gpu_devices {
                report.push_str(&format!("Device {}: {} {} ({:.1} GB)\n",
                    gpu.device_id,
                    gpu.vendor,
                    gpu.name,
                    gpu.memory_size as f64 / (1024.0 * 1024.0 * 1024.0)
                ));
                report.push_str(&format!("  Type: {:?}\n", gpu.device_type));
                report.push_str(&format!("  Compute Units: {}\n", gpu.max_compute_units));
                report.push_str(&format!("  Memory Bandwidth: {:.1} GB/s\n", gpu.memory_bandwidth()));
                if let Some(score) = profile.performance_characteristics.gpu_compute_scores.get(&gpu.device_id) {
                    report.push_str(&format!("  Compute Score: {:.1}\n", score));
                }
                report.push('\n');
            }
        }

        // Performance Recommendations
        report.push_str("=== Performance Recommendations ===\n");
        report.push_str(&self.generate_recommendations(profile));

        report
    }

    fn generate_recommendations(&self, profile: &HardwareProfile) -> String {
        let mut recommendations = String::new();

        // CPU recommendations
        if profile.cpu_info.cores >= 8 {
            recommendations.push_str("✓ Multi-core CPU detected - parallel operations will be efficient\n");
        } else {
            recommendations.push_str("⚠ Limited CPU cores - consider optimizing for single-threaded performance\n");
        }

        // SIMD recommendations
        match profile.cpu_info.features.simd_support {
            super::SIMDSupport::AVX512 | super::SIMDSupport::AVX2 => {
                recommendations.push_str("✓ Advanced SIMD support - vectorized operations will be highly efficient\n");
            }
            super::SIMDSupport::AVX | super::SIMDSupport::SSE4_2 => {
                recommendations.push_str("✓ Good SIMD support - vectorized operations will be efficient\n");
            }
            _ => {
                recommendations.push_str("⚠ Limited SIMD support - consider scalar optimizations\n");
            }
        }

        // Memory recommendations
        let memory_pressure = profile.memory_info.memory_pressure();
        if memory_pressure > 0.8 {
            recommendations.push_str("⚠ High memory pressure - consider reducing batch sizes\n");
        } else if memory_pressure < 0.5 {
            recommendations.push_str("✓ Plenty of available memory - can use larger batch sizes\n");
        }

        // GPU recommendations
        if profile.gpu_devices.is_empty() {
            recommendations.push_str("⚠ No GPU detected - computations will run on CPU only\n");
        } else {
            let best_gpu = profile.gpu_devices.iter()
                .filter(|gpu| gpu.is_available)
                .max_by_key(|gpu| gpu.memory_size);
            
            if let Some(gpu) = best_gpu {
                recommendations.push_str(&format!("✓ GPU available: {} - offload large computations to GPU\n", gpu.name));
                if gpu.memory_size < 4 * 1024 * 1024 * 1024 { // < 4GB
                    recommendations.push_str("⚠ Limited GPU memory - use smaller batch sizes for GPU operations\n");
                }
            }
        }

        recommendations
    }
}

impl SystemInfo {
    /// Detect system information
    pub fn detect() -> Self {
        Self {
            os_name: Self::get_os_name(),
            os_version: Self::get_os_version(),
            architecture: Self::get_architecture(),
            hostname: Self::get_hostname(),
            uptime: Self::get_uptime(),
            load_average: Self::get_load_average(),
        }
    }

    fn get_os_name() -> String {
        std::env::consts::OS.to_string()
    }

    fn get_os_version() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        return line.split('=').nth(1)
                            .unwrap_or("Unknown")
                            .trim_matches('"')
                            .to_string();
                    }
                }
            }
        }

        "Unknown".to_string()
    }

    fn get_architecture() -> String {
        std::env::consts::ARCH.to_string()
    }

    fn get_hostname() -> String {
        std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_string())
    }

    fn get_uptime() -> Duration {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/uptime") {
                if let Some(uptime_str) = content.split_whitespace().next() {
                    if let Ok(uptime_secs) = uptime_str.parse::<f64>() {
                        return Duration::from_secs_f64(uptime_secs);
                    }
                }
            }
        }

        Duration::ZERO
    }

    fn get_load_average() -> Option<(f64, f64, f64)> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                let parts: Vec<&str> = content.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let (Ok(load1), Ok(load5), Ok(load15)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                    ) {
                        return Some((load1, load5, load15));
                    }
                }
            }
        }

        None
    }
}

impl ThermalInfo {
    /// Detect thermal information
    pub fn detect() -> Self {
        Self {
            cpu_temperature: Self::get_cpu_temperature(),
            gpu_temperatures: Self::get_gpu_temperatures(),
            thermal_throttling_detected: false, // Would need more sophisticated detection
            cooling_capability: Self::assess_cooling_capability(),
        }
    }

    fn get_cpu_temperature() -> Option<f64> {
        #[cfg(target_os = "linux")]
        {
            // Try to read from thermal zones
            for i in 0..10 {
                let temp_path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
                if let Ok(temp_str) = std::fs::read_to_string(&temp_path) {
                    if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                        return Some(temp_millicelsius as f64 / 1000.0);
                    }
                }
            }
        }

        None
    }

    fn get_gpu_temperatures() -> HashMap<usize, f64> {
        HashMap::new() // Placeholder - would need GPU-specific APIs
    }

    fn assess_cooling_capability() -> CoolingCapability {
        // Simple heuristic based on system type
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/sys/class/dmi/id/chassis_type") {
                match content.trim() {
                    "3" => return CoolingCapability::Excellent, // Desktop
                    "9" | "10" => return CoolingCapability::Good, // Laptop
                    "30" | "31" => return CoolingCapability::Limited, // Tablet
                    _ => {}
                }
            }
        }

        CoolingCapability::Unknown
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profiler() {
        let profiler = HardwareProfiler::new();
        let profile = profiler.profile_system();
        
        assert!(profile.cpu_info.cores > 0);
        assert!(profile.memory_info.total_ram > 0);
        assert!(!profile.system_info.os_name.is_empty());
    }

    #[test]
    fn test_performance_report() {
        let profiler = HardwareProfiler::new();
        let profile = profiler.profile_system();
        let report = profiler.generate_report(&profile);
        
        assert!(report.contains("Hardware Profile Report"));
        assert!(report.contains("CPU Information"));
        assert!(report.contains("Memory Information"));
    }

    #[test]
    fn test_system_info_detection() {
        let system_info = SystemInfo::detect();
        assert!(!system_info.os_name.is_empty());
        assert!(!system_info.architecture.is_empty());
    }
}