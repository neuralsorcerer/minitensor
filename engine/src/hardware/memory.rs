// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use std::time::{Duration, Instant};

/// System memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_ram: usize,      // bytes
    pub available_ram: usize,  // bytes
    pub total_swap: usize,     // bytes
    pub available_swap: usize, // bytes
    pub page_size: usize,      // bytes
    pub bandwidth: MemoryBandwidth,
    pub cache_info: Vec<CacheInfo>,
}

/// Memory bandwidth measurements
#[derive(Debug, Clone)]
pub struct MemoryBandwidth {
    pub sequential_read: f64,  // GB/s
    pub sequential_write: f64, // GB/s
    pub random_read: f64,      // GB/s
    pub random_write: f64,     // GB/s
    pub copy_bandwidth: f64,   // GB/s
}

/// Cache hierarchy information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub level: u8,
    pub size: usize,      // bytes
    pub line_size: usize, // bytes
    pub associativity: usize,
    pub latency_cycles: Option<usize>,
    pub bandwidth: Option<f64>, // GB/s
}

impl MemoryInfo {
    /// Detect system memory information
    pub fn detect() -> Self {
        Self {
            total_ram: Self::get_total_ram(),
            available_ram: Self::get_available_ram(),
            total_swap: Self::get_total_swap(),
            available_swap: Self::get_available_swap(),
            page_size: Self::get_page_size(),
            bandwidth: MemoryBandwidth::benchmark(),
            cache_info: Self::detect_cache_hierarchy(),
        }
    }

    fn get_total_ram() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
                if let Ok(size_str) = String::from_utf8(output.stdout) {
                    if let Ok(size) = size_str.trim().parse::<usize>() {
                        return size;
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Windows implementation would use GetPhysicallyInstalledSystemMemory
            // For now, return a reasonable default
            return 8 * 1024 * 1024 * 1024; // 8GB default
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Fallback for unsupported platforms or if detection fails
            8 * 1024 * 1024 * 1024 // 8GB default
        }
    }

    fn get_available_ram() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback: assume 75% of total RAM is available
        (Self::get_total_ram() * 3) / 4
    }

    fn get_total_swap() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("SwapTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        0 // Default to no swap
    }

    fn get_available_swap() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("SwapFree:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        0 // Default to no swap
    }

    fn get_page_size() -> usize {
        #[cfg(unix)]
        {
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
        }
        #[cfg(not(unix))]
        {
            4096 // 4KB default page size
        }
    }

    fn detect_cache_hierarchy() -> Vec<CacheInfo> {
        let mut cache_info = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // Try to read cache information from sysfs
            for level in 1..=4 {
                let cache_path = format!("/sys/devices/system/cpu/cpu0/cache/index{}", level - 1);

                if let (Ok(size_str), Ok(line_size_str)) = (
                    std::fs::read_to_string(format!("{}/size", cache_path)),
                    std::fs::read_to_string(format!("{}/coherency_line_size", cache_path)),
                ) {
                    if let (Some(size), Ok(line_size)) = (
                        Self::parse_cache_size(&size_str),
                        line_size_str.trim().parse::<usize>(),
                    ) {
                        // Try to read associativity
                        let associativity = std::fs::read_to_string(format!(
                            "{}/ways_of_associativity",
                            cache_path
                        ))
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(8); // Default associativity

                        cache_info.push(CacheInfo {
                            level: level as u8,
                            size,
                            line_size,
                            associativity,
                            latency_cycles: Self::estimate_cache_latency(level as u8),
                            bandwidth: None, // Will be benchmarked separately if needed
                        });
                    }
                }
            }
        }

        // Fallback defaults if detection fails
        if cache_info.is_empty() {
            cache_info.extend([
                CacheInfo {
                    level: 1,
                    size: 32 * 1024, // 32KB L1
                    line_size: 64,
                    associativity: 8,
                    latency_cycles: Some(4),
                    bandwidth: None,
                },
                CacheInfo {
                    level: 2,
                    size: 256 * 1024, // 256KB L2
                    line_size: 64,
                    associativity: 8,
                    latency_cycles: Some(12),
                    bandwidth: None,
                },
                CacheInfo {
                    level: 3,
                    size: 8 * 1024 * 1024, // 8MB L3
                    line_size: 64,
                    associativity: 16,
                    latency_cycles: Some(40),
                    bandwidth: None,
                },
            ]);
        }

        cache_info
    }

    #[cfg(target_os = "linux")]
    fn parse_cache_size(size_str: &str) -> Option<usize> {
        let size_str = size_str.trim().to_uppercase();

        if let Some(kb_pos) = size_str.find('K') {
            if let Ok(kb) = size_str[..kb_pos].parse::<usize>() {
                return Some(kb * 1024);
            }
        }

        if let Some(mb_pos) = size_str.find('M') {
            if let Ok(mb) = size_str[..mb_pos].parse::<usize>() {
                return Some(mb * 1024 * 1024);
            }
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn estimate_cache_latency(level: u8) -> Option<usize> {
        match level {
            1 => Some(4),  // ~4 cycles for L1
            2 => Some(12), // ~12 cycles for L2
            3 => Some(40), // ~40 cycles for L3
            _ => None,
        }
    }

    /// Get the optimal buffer size for memory operations
    pub fn optimal_buffer_size(&self) -> usize {
        // Use L3 cache size if available, otherwise use a reasonable default
        self.cache_info
            .iter()
            .find(|cache| cache.level == 3)
            .map(|cache| cache.size / 2) // Use half of L3 cache
            .unwrap_or(2 * 1024 * 1024) // 2MB default
    }

    /// Check if there's enough memory for an allocation
    pub fn can_allocate(&self, size: usize) -> bool {
        size <= self.available_ram
    }

    /// Get memory pressure level (0.0 = no pressure, 1.0 = critical)
    pub fn memory_pressure(&self) -> f64 {
        let used_ram = self.total_ram - self.available_ram;
        used_ram as f64 / self.total_ram as f64
    }
}

impl MemoryBandwidth {
    /// Benchmark memory bandwidth
    pub fn benchmark() -> Self {
        let test_size = 64 * 1024 * 1024; // 64MB test buffer

        Self {
            sequential_read: Self::benchmark_sequential_read(test_size),
            sequential_write: Self::benchmark_sequential_write(test_size),
            random_read: Self::benchmark_random_read(test_size),
            random_write: Self::benchmark_random_write(test_size),
            copy_bandwidth: Self::benchmark_copy(test_size),
        }
    }

    fn benchmark_sequential_read(size: usize) -> f64 {
        let buffer = vec![0u8; size];
        let iterations = 10;
        let mut total_time = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();

            // Sequential read benchmark
            let mut sum = 0u64;
            for chunk in buffer.chunks(8) {
                sum += chunk.iter().map(|&b| b as u64).sum::<u64>();
            }

            // Prevent optimization
            std::hint::black_box(sum);

            total_time += start.elapsed();
        }

        let avg_time = total_time.as_secs_f64() / iterations as f64;
        let bytes_per_second = size as f64 / avg_time;
        bytes_per_second / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }

    fn benchmark_sequential_write(size: usize) -> f64 {
        let mut buffer = vec![0u8; size];
        let iterations = 10;
        let mut total_time = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();

            // Sequential write benchmark
            for (i, byte) in buffer.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            total_time += start.elapsed();
        }

        let avg_time = total_time.as_secs_f64() / iterations as f64;
        let bytes_per_second = size as f64 / avg_time;
        bytes_per_second / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }

    fn benchmark_random_read(size: usize) -> f64 {
        let buffer = vec![0u8; size];
        let iterations = 10;
        let indices: Vec<usize> = (0..size).step_by(64).collect(); // Cache line aligned
        let mut total_time = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();

            // Random read benchmark
            let mut sum = 0u64;
            for &idx in &indices {
                if idx < buffer.len() {
                    sum += buffer[idx] as u64;
                }
            }

            // Prevent optimization
            std::hint::black_box(sum);

            total_time += start.elapsed();
        }

        let avg_time = total_time.as_secs_f64() / iterations as f64;
        let bytes_accessed = indices.len() * 64; // Assume cache line reads
        let bytes_per_second = bytes_accessed as f64 / avg_time;
        bytes_per_second / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }

    fn benchmark_random_write(size: usize) -> f64 {
        let mut buffer = vec![0u8; size];
        let iterations = 10;
        let indices: Vec<usize> = (0..size).step_by(64).collect(); // Cache line aligned
        let mut total_time = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();

            // Random write benchmark
            for (i, &idx) in indices.iter().enumerate() {
                if idx < buffer.len() {
                    buffer[idx] = (i % 256) as u8;
                }
            }

            total_time += start.elapsed();
        }

        let avg_time = total_time.as_secs_f64() / iterations as f64;
        let bytes_accessed = indices.len() * 64; // Assume cache line writes
        let bytes_per_second = bytes_accessed as f64 / avg_time;
        bytes_per_second / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }

    fn benchmark_copy(size: usize) -> f64 {
        let src = vec![1u8; size];
        let mut dst = vec![0u8; size];
        let iterations = 10;
        let mut total_time = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();

            // Memory copy benchmark
            dst.copy_from_slice(&src);

            total_time += start.elapsed();
        }

        let avg_time = total_time.as_secs_f64() / iterations as f64;
        let bytes_per_second = (size * 2) as f64 / avg_time; // Read + Write
        bytes_per_second / (1024.0 * 1024.0 * 1024.0) // Convert to GB/s
    }

    /// Get the effective bandwidth for a given access pattern
    pub fn effective_bandwidth(&self, pattern: AccessPattern) -> f64 {
        match pattern {
            AccessPattern::SequentialRead => self.sequential_read,
            AccessPattern::SequentialWrite => self.sequential_write,
            AccessPattern::RandomRead => self.random_read,
            AccessPattern::RandomWrite => self.random_write,
            AccessPattern::Copy => self.copy_bandwidth,
            AccessPattern::Mixed => {
                // Weighted average for mixed workloads
                (self.sequential_read + self.sequential_write + self.copy_bandwidth) / 3.0
            }
        }
    }
}

/// Memory access patterns for bandwidth estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    SequentialRead,
    SequentialWrite,
    RandomRead,
    RandomWrite,
    Copy,
    Mixed,
}

impl CacheInfo {
    /// Calculate the number of cache lines for a given size
    pub fn cache_lines(&self, size: usize) -> usize {
        (size + self.line_size - 1) / self.line_size
    }

    /// Check if a data size fits in this cache level
    pub fn fits_in_cache(&self, size: usize) -> bool {
        size <= self.size
    }

    /// Estimate access latency for this cache level
    pub fn access_latency(&self) -> Duration {
        let cycles = self.latency_cycles.unwrap_or_else(|| match self.level {
            1 => 4,
            2 => 12,
            3 => 40,
            _ => 100,
        });

        // Assume 3GHz CPU for latency estimation
        let cpu_frequency = 3_000_000_000.0; // 3 GHz
        let seconds_per_cycle = 1.0 / cpu_frequency;
        Duration::from_secs_f64(cycles as f64 * seconds_per_cycle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_detection() {
        let memory_info = MemoryInfo::detect();
        assert!(memory_info.total_ram > 0);
        assert!(memory_info.page_size > 0);
        assert!(!memory_info.cache_info.is_empty());
    }

    #[test]
    fn test_memory_bandwidth_benchmark() {
        let bandwidth = MemoryBandwidth::benchmark();
        assert!(bandwidth.sequential_read > 0.0);
        assert!(bandwidth.random_read > 0.0);
        assert!(bandwidth.sequential_write > 0.0);
        assert!(bandwidth.random_write > 0.0);
        assert!(bandwidth.copy_bandwidth > 0.0);
    }

    #[test]
    fn test_cache_info() {
        let cache = CacheInfo {
            level: 1,
            size: 32 * 1024,
            line_size: 64,
            associativity: 8,
            latency_cycles: Some(4),
            bandwidth: None,
        };

        assert_eq!(cache.cache_lines(128), 2);
        assert!(cache.fits_in_cache(16 * 1024));
        assert!(!cache.fits_in_cache(64 * 1024));
    }

    #[test]
    fn test_memory_pressure() {
        let memory_info = MemoryInfo {
            total_ram: 8 * 1024 * 1024 * 1024,     // 8GB
            available_ram: 4 * 1024 * 1024 * 1024, // 4GB available
            total_swap: 0,
            available_swap: 0,
            page_size: 4096,
            bandwidth: MemoryBandwidth {
                sequential_read: 20.0,
                sequential_write: 15.0,
                random_read: 5.0,
                random_write: 3.0,
                copy_bandwidth: 18.0,
            },
            cache_info: Vec::new(),
        };

        assert_eq!(memory_info.memory_pressure(), 0.5); // 50% used
        assert!(memory_info.can_allocate(2 * 1024 * 1024 * 1024)); // 2GB
        assert!(!memory_info.can_allocate(6 * 1024 * 1024 * 1024)); // 6GB
    }
}
