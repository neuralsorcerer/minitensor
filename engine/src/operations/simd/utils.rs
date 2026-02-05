// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn simd_div_f64_scalar(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    for i in 0..lhs.len() {
        output[i] = if rhs[i] == 0.0 {
            f64::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }
    Ok(())
}

// x86_64 AVX2 f64 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_f64_avx2(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 4; // AVX2 processes 4 f64s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm256_add_pd(a, b);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sub_f64_avx2(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm256_sub_pd(a, b);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_mul_f64_avx2(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm256_mul_pd(a, b);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_div_f64_avx2(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 4;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm256_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm256_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm256_div_pd(a, b);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f64::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}

// x86_64 SSE f64 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_add_f64_sse(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2; // SSE processes 2 f64s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm_add_pd(a, b);
            _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_sub_f64_sse(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm_sub_pd(a, b);
            _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_mul_f64_sse(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm_mul_pd(a, b);
            _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_div_f64_sse(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = _mm_loadu_pd(lhs.as_ptr().add(i));
            let b = _mm_loadu_pd(rhs.as_ptr().add(i));
            let result = _mm_div_pd(a, b);
            _mm_storeu_pd(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f64::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}

// ARM NEON f64 implementations
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_add_f64_neon(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2; // NEON processes 2 f64s at once

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let result = vaddq_f64(a, b);
            vst1q_f64(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] + rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_sub_f64_neon(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let result = vsubq_f64(a, b);
            vst1q_f64(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] - rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_mul_f64_neon(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let result = vmulq_f64(a, b);
            vst1q_f64(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = lhs[i] * rhs[i];
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_div_f64_neon(lhs: &[f64], rhs: &[f64], output: &mut [f64]) -> Result<()> {
    const SIMD_WIDTH: usize = 2;

    let len = lhs.len();
    let simd_len = len - (len % SIMD_WIDTH);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a = vld1q_f64(lhs.as_ptr().add(i));
            let b = vld1q_f64(rhs.as_ptr().add(i));
            let result = vdivq_f64(a, b);
            vst1q_f64(output.as_mut_ptr().add(i), result);
        }
    }

    for i in simd_len..len {
        output[i] = if rhs[i] == 0.0 {
            f64::INFINITY
        } else {
            lhs[i] / rhs[i]
        };
    }

    Ok(())
}
