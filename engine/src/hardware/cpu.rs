// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

/// CPU information and capabilities
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model_name: String,
    pub vendor: String,
    pub cores: usize,
    pub threads: usize,
    pub base_frequency: Option<f64>, // MHz
    pub max_frequency: Option<f64>,  // MHz
    pub features: CpuFeatures,
    pub cache_info: Vec<CacheLevel>,
}

/// CPU feature flags and capabilities
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub simd_support: SIMDSupport,
    pub has_fma: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_neon: bool, // ARM NEON
    pub has_sve: bool,  // ARM SVE
}

/// SIMD instruction set support levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SIMDSupport {
    None,
    SSE,
    SSE2,
    SSE3,
    SSE4_1,
    SSE4_2,
    AVX,
    AVX2,
    AVX512,
    NEON,
    SVE,
}

/// CPU cache level information
#[derive(Debug, Clone)]
pub struct CacheLevel {
    pub level: u8,
    pub cache_type: CacheType,
    pub size: usize,      // bytes
    pub line_size: usize, // bytes
    pub associativity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheType {
    Data,
    Instruction,
    Unified,
}

impl CpuInfo {
    /// Detect CPU information and capabilities
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::detect_generic()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        let mut cpu_info = Self::detect_generic();

        // Use CPUID to detect features
        if is_x86_feature_detected!("sse") {
            cpu_info.features.has_sse = true;
        }
        if is_x86_feature_detected!("sse2") {
            cpu_info.features.has_sse2 = true;
        }
        if is_x86_feature_detected!("sse3") {
            cpu_info.features.has_sse3 = true;
        }
        if is_x86_feature_detected!("sse4.1") {
            cpu_info.features.has_sse4_1 = true;
        }
        if is_x86_feature_detected!("sse4.2") {
            cpu_info.features.has_sse4_2 = true;
        }
        if is_x86_feature_detected!("avx") {
            cpu_info.features.has_avx = true;
        }
        if is_x86_feature_detected!("avx2") {
            cpu_info.features.has_avx2 = true;
        }
        if is_x86_feature_detected!("avx512f") {
            cpu_info.features.has_avx512f = true;
        }
        if is_x86_feature_detected!("fma") {
            cpu_info.features.has_fma = true;
        }

        // Determine highest SIMD support
        cpu_info.features.simd_support = if cpu_info.features.has_avx512f {
            SIMDSupport::AVX512
        } else if cpu_info.features.has_avx2 {
            SIMDSupport::AVX2
        } else if cpu_info.features.has_avx {
            SIMDSupport::AVX
        } else if cpu_info.features.has_sse4_2 {
            SIMDSupport::SSE4_2
        } else if cpu_info.features.has_sse4_1 {
            SIMDSupport::SSE4_1
        } else if cpu_info.features.has_sse3 {
            SIMDSupport::SSE3
        } else if cpu_info.features.has_sse2 {
            SIMDSupport::SSE2
        } else if cpu_info.features.has_sse {
            SIMDSupport::SSE
        } else {
            SIMDSupport::None
        };

        cpu_info
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        let mut cpu_info = Self::detect_generic();

        // ARM NEON is standard on AArch64
        cpu_info.features.has_neon = true;
        cpu_info.features.simd_support = SIMDSupport::NEON;

        // Check for SVE support (if available in std)
        #[cfg(target_feature = "sve")]
        {
            cpu_info.features.has_sve = true;
            cpu_info.features.simd_support = SIMDSupport::SVE;
        }

        cpu_info
    }

    fn detect_generic() -> Self {
        let cores = num_cpus::get();
        let threads = num_cpus::get(); // Assume no hyperthreading for generic case

        Self {
            model_name: Self::get_cpu_model(),
            vendor: Self::get_cpu_vendor(),
            cores,
            threads,
            base_frequency: None,
            max_frequency: None,
            features: CpuFeatures::default(),
            cache_info: Self::detect_cache_info(),
        }
    }

    fn get_cpu_model() -> String {
        // Try to read from /proc/cpuinfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            return name.trim().to_string();
                        }
                    }
                }
            }
        }

        // Fallback
        "Unknown CPU".to_string()
    }

    fn get_cpu_vendor() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("vendor_id") {
                        if let Some(vendor) = line.split(':').nth(1) {
                            return vendor.trim().to_string();
                        }
                    }
                }
            }
        }

        "Unknown".to_string()
    }

    fn detect_cache_info() -> Vec<CacheLevel> {
        let mut cache_levels = Vec::new();

        // Try to detect cache sizes (Linux-specific for now)
        #[cfg(target_os = "linux")]
        {
            // L1 data cache
            if let Ok(size_str) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size")
            {
                if let Some(size) = Self::parse_cache_size(&size_str) {
                    cache_levels.push(CacheLevel {
                        level: 1,
                        cache_type: CacheType::Data,
                        size,
                        line_size: 64,    // Common default
                        associativity: 8, // Common default
                    });
                }
            }

            // L1 instruction cache
            if let Ok(size_str) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index1/size")
            {
                if let Some(size) = Self::parse_cache_size(&size_str) {
                    cache_levels.push(CacheLevel {
                        level: 1,
                        cache_type: CacheType::Instruction,
                        size,
                        line_size: 64,
                        associativity: 8,
                    });
                }
            }

            // L2 cache
            if let Ok(size_str) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            {
                if let Some(size) = Self::parse_cache_size(&size_str) {
                    cache_levels.push(CacheLevel {
                        level: 2,
                        cache_type: CacheType::Unified,
                        size,
                        line_size: 64,
                        associativity: 8,
                    });
                }
            }

            // L3 cache
            if let Ok(size_str) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size")
            {
                if let Some(size) = Self::parse_cache_size(&size_str) {
                    cache_levels.push(CacheLevel {
                        level: 3,
                        cache_type: CacheType::Unified,
                        size,
                        line_size: 64,
                        associativity: 16,
                    });
                }
            }
        }

        // Fallback defaults if detection fails
        if cache_levels.is_empty() {
            cache_levels.extend([
                CacheLevel {
                    level: 1,
                    cache_type: CacheType::Data,
                    size: 32 * 1024, // 32KB
                    line_size: 64,
                    associativity: 8,
                },
                CacheLevel {
                    level: 2,
                    cache_type: CacheType::Unified,
                    size: 256 * 1024, // 256KB
                    line_size: 64,
                    associativity: 8,
                },
                CacheLevel {
                    level: 3,
                    cache_type: CacheType::Unified,
                    size: 8 * 1024 * 1024, // 8MB
                    line_size: 64,
                    associativity: 16,
                },
            ]);
        }

        cache_levels
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

    /// Get the optimal number of threads for parallel operations
    pub fn optimal_thread_count(&self) -> usize {
        // Use physical cores for CPU-bound tasks
        self.cores
    }

    /// Check if CPU supports a specific SIMD instruction set
    pub fn supports_simd(&self, simd: SIMDSupport) -> bool {
        match simd {
            SIMDSupport::None => true,
            SIMDSupport::SSE => self.features.has_sse,
            SIMDSupport::SSE2 => self.features.has_sse2,
            SIMDSupport::SSE3 => self.features.has_sse3,
            SIMDSupport::SSE4_1 => self.features.has_sse4_1,
            SIMDSupport::SSE4_2 => self.features.has_sse4_2,
            SIMDSupport::AVX => self.features.has_avx,
            SIMDSupport::AVX2 => self.features.has_avx2,
            SIMDSupport::AVX512 => self.features.has_avx512f,
            SIMDSupport::NEON => self.features.has_neon,
            SIMDSupport::SVE => self.features.has_sve,
        }
    }
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            simd_support: SIMDSupport::None,
            has_fma: false,
            has_avx: false,
            has_avx2: false,
            has_avx512f: false,
            has_sse: false,
            has_sse2: false,
            has_sse3: false,
            has_sse4_1: false,
            has_sse4_2: false,
            has_neon: false,
            has_sve: false,
        }
    }
}

impl SIMDSupport {
    /// Get the vector width in bytes for this SIMD instruction set
    pub fn vector_width(&self) -> usize {
        match self {
            SIMDSupport::None => 1,
            SIMDSupport::SSE
            | SIMDSupport::SSE2
            | SIMDSupport::SSE3
            | SIMDSupport::SSE4_1
            | SIMDSupport::SSE4_2 => 16, // 128-bit
            SIMDSupport::AVX | SIMDSupport::AVX2 => 32, // 256-bit
            SIMDSupport::AVX512 => 64,                  // 512-bit
            SIMDSupport::NEON => 16,                    // 128-bit
            SIMDSupport::SVE => 16,                     // Variable, but assume 128-bit minimum
        }
    }

    /// Get the number of f32 elements that fit in one SIMD register
    pub fn f32_lanes(&self) -> usize {
        self.vector_width() / 4
    }

    /// Get the number of f64 elements that fit in one SIMD register
    pub fn f64_lanes(&self) -> usize {
        self.vector_width() / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let cpu_info = CpuInfo::detect();
        assert!(cpu_info.cores > 0);
        assert!(cpu_info.threads > 0);
        assert!(!cpu_info.model_name.is_empty());
    }

    #[test]
    fn test_simd_vector_width() {
        assert_eq!(SIMDSupport::SSE.vector_width(), 16);
        assert_eq!(SIMDSupport::AVX2.vector_width(), 32);
        assert_eq!(SIMDSupport::AVX512.vector_width(), 64);
    }

    #[test]
    fn test_simd_lanes() {
        assert_eq!(SIMDSupport::AVX2.f32_lanes(), 8);
        assert_eq!(SIMDSupport::AVX2.f64_lanes(), 4);
    }
}
