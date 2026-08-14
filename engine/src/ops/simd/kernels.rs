// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::Shape,
};
use std::mem::MaybeUninit;

/// SIMD capabilities detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    /// 256-bit float arithmetic (`vaddps`/`vaddpd`). This, not [`Self::avx2`],
    /// is what the element-wise float kernels need: AVX2 extends the *integer*
    /// instructions, and a Sandy/Ivy Bridge machine has the wide float
    /// registers without it.
    pub avx: bool,
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
            avx: is_x86_feature_detected!("avx"),
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            sse4_1: is_x86_feature_detected!("sse4.1"),
            #[cfg(not(target_arch = "x86_64"))]
            avx: false,
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

/// Generates one element-wise binary kernel: the loop, one compilation of it
/// per instruction set, and the runtime pick between them.
///
/// Every kernel here is the same loop — `out[i] = lhs[i] OP rhs[i]` — and that
/// is a loop LLVM vectorizes exactly, provided it is told which registers it
/// may use. So the loop is written once and compiled twice, the same way
/// `ops::simd::transcendental` handles its far more intricate kernels.
///
/// This replaces four hand-written intrinsic bodies per operation (AVX2, SSE,
/// NEON, scalar) per dtype — 24 functions of `_mm256_loadu_ps`/`step_by`
/// boilerplate that between them said nothing the operator did not. Writing
/// them out cost more than the duplication:
///
/// * **Three of the four paths were never distinguishable from the fallback.**
///   SSE2 is baseline on every x86-64 target, so the "scalar" body already
///   compiled to `addps`/`addpd`; the `sse4.1` variants emitted the identical
///   instructions behind a runtime check, since SSE4.1 adds nothing to float
///   arithmetic. NEON is likewise baseline on aarch64, so the NEON bodies were
///   the fallback with extra steps.
/// * **The hand-written loops were the slower ones.** One vector operation per
///   iteration with no unrolling, and a scalar remainder loop that kept its
///   bounds checks. LLVM unrolls the same loop four ways and vectorizes the
///   remainder, which is worth 1.4x-1.5x on cache-resident data (float32 add,
///   4096 elements: 0.94us -> 0.67us).
///
/// # Why there is no AVX-512 tier
///
/// A third compilation is one line here, so it was measured rather than
/// assumed — and 512-bit registers lost at every size on the AVX-512 machine
/// available, by 26% on cache-resident data and 34% when memory-bound
/// (float32 add, us, one process so the ISA is the only variable):
///
/// ```text
///        N     SSE2      AVX   AVX-512
///     4096    0.739    0.485     0.609
///    65536   12.154    9.962    11.777
///  4194304 4035.903 4333.065  5427.977
/// ```
///
/// Which is the ordinary shape of it: these kernels are one arithmetic
/// operation per two loads and a store, so they saturate on memory long before
/// they saturate on vector width, and the frequency these parts drop to when
/// 512-bit code issues is not repaid. Note the last row — past the caches even
/// AVX loses to the baseline, which is what makes the parallel split in
/// `ops::kernels::binary` matter far more here than the register width does.
macro_rules! binary_elementwise {
    ($(#[$meta:meta])* $entry:ident, $core:ident, $ty:ty, $op:tt) => {
        /// The dispatching loop, with the length agreement taken on trust.
        ///
        /// `out.len()` sets the length and both inputs are reborrowed to it, so
        /// a caller that passes a short input panics on the reborrow rather
        /// than leaving the output partly written. [`$entry`] is the checked
        /// public form; this one exists so the blocked driver in
        /// `ops::kernels::binary` — which slices all three buffers itself, and
        /// so cannot get the lengths wrong — does not have to answer a `Result`
        /// it can never receive.
        #[inline]
        pub(crate) fn $core(lhs: &[$ty], rhs: &[$ty], out: &mut [MaybeUninit<$ty>]) {
            /// The loop. `#[inline(always)]` is what makes the wrapper below a
            /// second compilation rather than a call to this one: inlining into
            /// a `#[target_feature]` function rebuilds the body with that
            /// function's registers available.
            #[inline(always)]
            fn body(lhs: &[$ty], rhs: &[$ty], out: &mut [MaybeUninit<$ty>]) {
                // Reborrowing both inputs at the output length is what removes
                // the bounds checks: it gives LLVM one length for all three
                // slices, so nothing in the loop can trap.
                let n = out.len();
                let (lhs, rhs) = (&lhs[..n], &rhs[..n]);
                for i in 0..n {
                    out[i].write(lhs[i] $op rhs[i]);
                }
            }

            // `avx`, not `avx2`: these are float operations, and 256-bit
            // `vaddps`/`vaddpd` arrived with AVX. AVX2 is an integer extension,
            // so gating on it would skip the machines that have the registers
            // but not the integer instructions.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx")]
            fn body_avx(lhs: &[$ty], rhs: &[$ty], out: &mut [MaybeUninit<$ty>]) {
                body(lhs, rhs, out)
            }

            #[cfg(target_arch = "x86_64")]
            if simd_capabilities().avx {
                // SAFETY: `detect` confirmed avx on this CPU.
                unsafe { body_avx(lhs, rhs, out) };
                return;
            }

            // Baseline: SSE2 on x86-64, NEON on aarch64 — both already
            // vectorize this loop, which is why the ISA-specific bodies for
            // them were never anything but this one.
            body(lhs, rhs, out)
        }

        $(#[$meta])*
        pub fn $entry(lhs: &[$ty], rhs: &[$ty], output: &mut [MaybeUninit<$ty>]) -> Result<()> {
            if lhs.len() != rhs.len() || lhs.len() != output.len() {
                return Err(MinitensorError::invalid_operation(
                    "Array lengths must match for SIMD operations",
                ));
            }
            $core(lhs, rhs, output);
            Ok(())
        }
    };
}

binary_elementwise!(
    /// Element-wise `lhs + rhs` for equal-length f32 slices.
    ///
    /// The binary SIMD entry points write into `MaybeUninit` output so freshly
    /// allocated (never zeroed) buffers can be used directly; on success every
    /// element of `output` has been written. See `ops::map` for the allocation
    /// pattern.
    simd_add_f32, add_f32_blocks, f32, +
);
binary_elementwise!(
    /// Element-wise `lhs - rhs` for equal-length f32 slices.
    simd_sub_f32, sub_f32_blocks, f32, -
);
binary_elementwise!(
    /// Element-wise `lhs * rhs` for equal-length f32 slices.
    simd_mul_f32, mul_f32_blocks, f32, *
);
binary_elementwise!(
    /// Element-wise `lhs / rhs` for equal-length f32 slices. IEEE semantics:
    /// a zero divisor yields ±inf, or NaN for `0 / 0`.
    simd_div_f32, div_f32_blocks, f32, /
);
binary_elementwise!(
    /// Element-wise `lhs + rhs` for equal-length f64 slices.
    simd_add_f64, add_f64_blocks, f64, +
);
binary_elementwise!(
    /// Element-wise `lhs - rhs` for equal-length f64 slices.
    simd_sub_f64, sub_f64_blocks, f64, -
);
binary_elementwise!(
    /// Element-wise `lhs * rhs` for equal-length f64 slices.
    simd_mul_f64, mul_f64_blocks, f64, *
);
binary_elementwise!(
    /// Element-wise `lhs / rhs` for equal-length f64 slices. IEEE semantics:
    /// a zero divisor yields ±inf, or NaN for `0 / 0`.
    simd_div_f64, div_f64_blocks, f64, /
);

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

    /// Run one of the binary SIMD entry points into a fresh output vector.
    fn run(
        f: impl Fn(&[f32], &[f32], &mut [MaybeUninit<f32>]) -> Result<()>,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<Vec<f32>> {
        // SAFETY: the SIMD entry points initialize every output element on Ok.
        unsafe { crate::ops::map::build_vec_with(lhs.len(), |out| f(lhs, rhs, out)) }
    }

    /// f64 variant of [`run`].
    fn run64(
        f: impl Fn(&[f64], &[f64], &mut [MaybeUninit<f64>]) -> Result<()>,
        lhs: &[f64],
        rhs: &[f64],
    ) -> Result<Vec<f64>> {
        // SAFETY: the SIMD entry points initialize every output element on Ok.
        unsafe { crate::ops::map::build_vec_with(lhs.len(), |out| f(lhs, rhs, out)) }
    }

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

        let result = run(simd_add_f32, &a, &b).unwrap();

        for i in 0..8 {
            assert_eq!(result[i], 9.0);
        }
    }

    #[test]
    fn test_simd_mul_f32() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];

        let result = run(simd_mul_f32, &a, &b).unwrap();

        assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_simd_div_f32() {
        let a = vec![12.0, 15.0, 20.0, 24.0];
        let b = vec![3.0, 5.0, 4.0, 6.0];

        let result = run(simd_div_f32, &a, &b).unwrap();

        assert_eq!(result, vec![4.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_simd_div_by_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 2.0];

        let result = run(simd_div_f32, &a, &b).unwrap();

        assert_eq!(result[0], f32::INFINITY);
        assert_eq!(result[1], 1.0);
    }

    #[test]
    fn test_simd_div_by_zero_ieee_semantics_including_tail() {
        let len = 19;
        let a: Vec<f32> = (0..len)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let b = vec![0.0_f32; len];
        let result = run(simd_div_f32, &a, &b).unwrap();
        for i in 0..len {
            match i % 3 {
                0 => assert_eq!(result[i], f32::NEG_INFINITY, "index {i}"),
                1 => assert!(result[i].is_nan(), "index {i}"),
                _ => assert_eq!(result[i], f32::INFINITY, "index {i}"),
            }
        }

        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let b64 = vec![0.0_f64; len];
        let result64 = run64(simd_div_f64, &a64, &b64).unwrap();
        for i in 0..len {
            match i % 3 {
                0 => assert_eq!(result64[i], f64::NEG_INFINITY, "index {i}"),
                1 => assert!(result64[i].is_nan(), "index {i}"),
                _ => assert_eq!(result64[i], f64::INFINITY, "index {i}"),
            }
        }
    }

    #[test]
    fn test_simd_f32_length_mismatch_errors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0];

        let err = run(simd_add_f32, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run(simd_sub_f32, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run(simd_mul_f32, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run(simd_div_f32, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );
    }

    #[test]
    fn test_simd_f64_all_ops_with_remainder_and_division_by_zero() {
        let a = vec![10.0_f64, -9.0, 8.0, -7.0, 6.0];
        let b = vec![2.0_f64, -3.0, 4.0, -7.0, 0.0];

        let out = run64(simd_add_f64, &a, &b).unwrap();
        assert_eq!(out, vec![12.0, -12.0, 12.0, -14.0, 6.0]);

        let out = run64(simd_sub_f64, &a, &b).unwrap();
        assert_eq!(out, vec![8.0, -6.0, 4.0, 0.0, 6.0]);

        let out = run64(simd_mul_f64, &a, &b).unwrap();
        assert_eq!(out, vec![20.0, 27.0, 32.0, 49.0, 0.0]);

        let out = run64(simd_div_f64, &a, &b).unwrap();
        assert_eq!(out[..4], [5.0, 3.0, 2.0, 1.0]);
        assert_eq!(out[4], f64::INFINITY);
    }

    #[test]
    fn test_simd_f64_length_mismatch_errors() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0];

        let err = run64(simd_add_f64, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run64(simd_sub_f64, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run64(simd_mul_f64, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );

        let err = run64(simd_div_f64, &a, &b).unwrap_err();
        assert!(
            err.to_string()
                .contains("Array lengths must match for SIMD operations")
        );
    }

    #[test]
    fn test_can_use_simd_fast_path_shape_conditions() {
        let same = Shape::new(vec![2, 8]);
        let different = Shape::new(vec![4, 4]);
        let too_small = Shape::new(vec![2, 4]);

        assert!(can_use_simd_fast_path(&same, &same, &same));
        assert!(!can_use_simd_fast_path(&same, &different, &same));
        assert!(!can_use_simd_fast_path(&same, &same, &different));
        assert!(!can_use_simd_fast_path(&too_small, &too_small, &too_small));
    }
}
