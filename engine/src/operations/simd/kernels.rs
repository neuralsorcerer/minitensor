// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::Shape,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub avx2: bool,
    pub avx512: bool,
    pub sse4_1: bool,
    pub neon: bool,
    pub sve: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            sse4_1: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,
            #[cfg(not(target_arch = "x86_64"))]
            avx512: false,
            #[cfg(not(target_arch = "x86_64"))]
            sse4_1: false,

            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "aarch64")]
            sve: std::arch::is_aarch64_feature_detected!("sve"),
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
            #[cfg(not(target_arch = "aarch64"))]
            sve: false,
        }
    }
}

/// Global SIMD capabilities (detected once at startup)
static SIMD_CAPS: std::sync::OnceLock<SimdCapabilities> = std::sync::OnceLock::new();

/// Get the detected SIMD capabilities
pub fn simd_capabilities() -> SimdCapabilities {
    *SIMD_CAPS.get_or_init(SimdCapabilities::detect)
}

/// SIMD-optimized element-wise addition for f32 arrays
pub fn simd_add_f32(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_add_f32_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_add_f32_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_add_f32_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_add_f32_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise subtraction for f32 arrays
pub fn simd_sub_f32(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_sub_f32_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_sub_f32_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_sub_f32_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_sub_f32_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise multiplication for f32 arrays
pub fn simd_mul_f32(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_mul_f32_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_mul_f32_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_mul_f32_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_mul_f32_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise division for f32 arrays
pub fn simd_div_f32(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_div_f32_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_div_f32_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_div_f32_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_div_f32_scalar(lhs, rhs, output)
}

// Scalar fallback implementations
fn simd_add_f32_scalar(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] + rhs[i];
    }
    Ok(())
}

/// Unrolled sum for f32 slices to leverage auto-vectorization
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    let mut sums = [0f32; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
        sums[3] += chunk[3];
        sums[4] += chunk[4];
        sums[5] += chunk[5];
        sums[6] += chunk[6];
        sums[7] += chunk[7];
    }
    let mut total: f32 = sums.iter().sum();
    total += rem.iter().copied().sum::<f32>();
    total
}

/// Unrolled sum for f64 slices to leverage auto-vectorization
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    let mut sums = [0f64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
        sums[3] += chunk[3];
    }
    let mut total: f64 = sums.iter().sum();
    total += rem.iter().copied().sum::<f64>();
    total
}

/// Unrolled sum for i32 slices to leverage auto-vectorization
pub fn simd_sum_i32(data: &[i32]) -> i32 {
    let mut sums = [0i32; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
        sums[3] += chunk[3];
        sums[4] += chunk[4];
        sums[5] += chunk[5];
        sums[6] += chunk[6];
        sums[7] += chunk[7];
    }
    let mut total: i32 = sums.iter().sum();
    total += rem.iter().copied().sum::<i32>();
    total
}

/// Unrolled sum for i64 slices to leverage auto-vectorization
pub fn simd_sum_i64(data: &[i64]) -> i64 {
    let mut sums = [0i64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
        sums[3] += chunk[3];
    }
    let mut total: i64 = sums.iter().sum();
    total += rem.iter().copied().sum::<i64>();
    total
}

/// Unrolled product for f32 slices to leverage auto-vectorization
pub fn simd_prod_f32(data: &[f32]) -> f32 {
    let mut prods = [1f32; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] *= chunk[0];
        prods[1] *= chunk[1];
        prods[2] *= chunk[2];
        prods[3] *= chunk[3];
        prods[4] *= chunk[4];
        prods[5] *= chunk[5];
        prods[6] *= chunk[6];
        prods[7] *= chunk[7];
    }
    let mut total: f32 = prods.iter().product();
    total *= rem.iter().copied().product::<f32>();
    total
}

/// Unrolled product for f64 slices to leverage auto-vectorization
pub fn simd_prod_f64(data: &[f64]) -> f64 {
    let mut prods = [1f64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] *= chunk[0];
        prods[1] *= chunk[1];
        prods[2] *= chunk[2];
        prods[3] *= chunk[3];
    }
    let mut total: f64 = prods.iter().product();
    total *= rem.iter().copied().product::<f64>();
    total
}

/// Unrolled product for i32 slices to leverage auto-vectorization
pub fn simd_prod_i32(data: &[i32]) -> i32 {
    let mut prods = [1i32; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] *= chunk[0];
        prods[1] *= chunk[1];
        prods[2] *= chunk[2];
        prods[3] *= chunk[3];
        prods[4] *= chunk[4];
        prods[5] *= chunk[5];
        prods[6] *= chunk[6];
        prods[7] *= chunk[7];
    }
    let mut total: i32 = prods.iter().product();
    total *= rem.iter().copied().product::<i32>();
    total
}

/// Unrolled product for i64 slices to leverage auto-vectorization
pub fn simd_prod_i64(data: &[i64]) -> i64 {
    let mut prods = [1i64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] *= chunk[0];
        prods[1] *= chunk[1];
        prods[2] *= chunk[2];
        prods[3] *= chunk[3];
    }
    let mut total: i64 = prods.iter().product();
    total *= rem.iter().copied().product::<i64>();
    total
}

fn simd_sub_f32_scalar(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] - rhs[i];
    }
    Ok(())
}

fn simd_mul_f32_scalar(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] * rhs[i];
    }
    Ok(())
}

fn simd_div_f32_scalar(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = if rhs[i] == 0.0 {
            f32::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }
    Ok(())
}

// x86_64 AVX2 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_f32_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 8; // AVX2 processes 8 f32s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    // Process SIMD_WIDTH elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm256_add_ps(a, b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sub_f32_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 8;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm256_sub_ps(a, b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_mul_f32_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 8;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_div_f32_avx2(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 8;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm256_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm256_div_ps(a, b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f32::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}

// x86_64 SSE implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_add_f32_sse(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4; // SSE processes 4 f32s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm_add_ps(a, b);
            _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_sub_f32_sse(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm_sub_ps(a, b);
            _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_mul_f32_sse(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm_mul_ps(a, b);
            _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_div_f32_sse(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_ps(lhs.as_ptr().add(i));
            let b = _mm_loadu_ps(rhs.as_ptr().add(i));
            let result = _mm_div_ps(a, b);
            _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f32::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}

// ARM NEON implementations
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_add_f32_neon(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4; // NEON processes 4 f32s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f32(lhs.as_ptr().add(i));
            let b = vld1q_f32(rhs.as_ptr().add(i));
            let result = vaddq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_sub_f32_neon(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f32(lhs.as_ptr().add(i));
            let b = vld1q_f32(rhs.as_ptr().add(i));
            let result = vsubq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_mul_f32_neon(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f32(lhs.as_ptr().add(i));
            let b = vld1q_f32(rhs.as_ptr().add(i));
            let result = vmulq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_div_f32_neon(lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f32(lhs.as_ptr().add(i));
            let b = vld1q_f32(rhs.as_ptr().add(i));
            let result = vdivq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f32::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}

/// Check if two tensors can use optimized SIMD operations (same shape, contiguous)
pub fn can_use_simd_fast_path(lhs_shape: &Shape, rhs_shape: &Shape, output_shape: &Shape) -> bool {
    // For now, only optimize when all shapes are identical (no broadcasting)
    // This ensures contiguous memory access patterns optimal for SIMD
    lhs_shape.dims() == rhs_shape.dims()
        && lhs_shape.dims() == output_shape.dims()
        && lhs_shape.numel() >= 16 // Only use SIMD for reasonably sized arrays
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = simd_capabilities();
        // Just ensure it doesn't panic and returns something reasonable
        println!("SIMD capabilities: {:?}", caps);
    }

    #[test]
    fn test_simd_add_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];

        simd_add_f32(&a, &b, &mut result).unwrap();

        for i in 0..8 {
            assert_eq!(result[i], 9.0);
        }
    }

    #[test]
    fn test_simd_mul_f32() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 4];

        simd_mul_f32(&a, &b, &mut result).unwrap();

        assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_simd_div_f32() {
        let a = vec![12.0, 15.0, 20.0, 24.0];
        let b = vec![3.0, 5.0, 4.0, 6.0];
        let mut result = vec![0.0; 4];

        simd_div_f32(&a, &b, &mut result).unwrap();

        assert_eq!(result, vec![4.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_simd_div_by_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 2.0];
        let mut result = vec![0.0; 2];

        simd_div_f32(&a, &b, &mut result).unwrap();

        assert_eq!(result[0], f32::INFINITY);
        assert_eq!(result[1], 1.0);
    }
}

/// SIMD-optimized element-wise addition for f64 arrays
pub fn simd_add_f64(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_add_f64_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_add_f64_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_add_f64_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_add_f64_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise subtraction for f64 arrays
pub fn simd_sub_f64(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_sub_f64_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_sub_f64_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_sub_f64_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_sub_f64_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise multiplication for f64 arrays
pub fn simd_mul_f64(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_mul_f64_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_mul_f64_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_mul_f64_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_mul_f64_scalar(lhs, rhs, output)
}

/// SIMD-optimized element-wise division for f64 arrays
pub fn simd_div_f64(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() != output.len() {
        return Err(MinitensorError::invalid_operation(
            "Array lengths must match for SIMD operations",
        ));
    }

    let caps = simd_capabilities();

    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            return unsafe { simd_div_f64_avx2(lhs, rhs, output) };
        } else if caps.sse4_1 {
            return unsafe { simd_div_f64_sse(lhs, rhs, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon {
            return unsafe { simd_div_f64_neon(lhs, rhs, output) };
        }
    }

    // Fallback to scalar implementation
    simd_div_f64_scalar(lhs, rhs, output)
}

// f64 scalar fallback implementations
fn simd_add_f64_scalar(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] + rhs[i];
    }
    Ok(())
}

fn simd_sub_f64_scalar(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] - rhs[i];
    }
    Ok(())
}

fn simd_mul_f64_scalar(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = lhs[i] * rhs[i];
    }
    Ok(())
}
