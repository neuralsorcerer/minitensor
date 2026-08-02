// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Vectorized transcendental kernels.
//!
//! `tanh` is the one transcendental this engine leans on everywhere -- every
//! RNN and LSTM cell, and most small MLPs -- and it was the slowest thing in
//! the elementwise surface: a `libm` call per element, which no amount of
//! rayon parallelism can vectorize away. Against NumPy (which ships its own
//! SIMD `tanh`) float32 measured 11.8x slower at a million elements and 31x at
//! 4096, *after* spreading the work over four cores.
//!
//! The fix is to write the routine so that LLVM can vectorize it, then compile
//! it several times over -- once per instruction set -- and pick at runtime.
//! That requires the whole loop to live inside the multiversioned function, so
//! the entry point here takes a block rather than an element (see
//! `ops::map::unary_map_blocks_threshold`).
//!
//! # What it computes
//!
//! `tanh(x) = u / (u + 2)` where `u = expm1(2x)`, evaluated in float64 and
//! rounded once to float32. Writing it through `expm1` rather than the more
//! obvious `1 - 2/(exp(2x) + 1)` is what makes it accurate near zero: the
//! latter cancels catastrophically as `x -> 0`, while `u/(u+2)` has `u ~ 2x`
//! over a denominator near 2 and stays well conditioned. Relative error in `u`
//! reaches the result damped by `2/(u+2) <= 1`, so nothing amplifies.
//!
//! `expm1(t)` uses the textbook reduction `t = n*ln2 + r`, `|r| <= ln2/2`,
//! recombined as
//!
//! ```text
//!     expm1(t) = 2^n * p + (2^n - 1),   p = exp(r) - 1
//! ```
//!
//! which is the same expression for every `n` and cancels for none of them:
//! at `n = 0` it collapses to `p`, and for `n != 0` the `2^n - 1` term is
//! bounded away from zero exactly where `p` is small. `2^n - 1` is exact for
//! the `|n| <= 29` this range produces.
//!
//! Three details are what make it fast rather than merely correct:
//!
//! * **Rounding without `roundpd`.** `f64::round` is round-half-away-from-zero,
//!   which x86 cannot do in one instruction, so LLVM scalarizes the loop around
//!   it. Adding `MAGIC` instead forces the mantissa to hold the rounded
//!   integer, and the same bits then build `2^n` by a shift -- no float/int
//!   conversion either, which would otherwise scalarize again on Rust's
//!   saturating cast.
//! * **FMA, explicitly.** Rust never contracts `a*b + c` into an FMA on its
//!   own. The polynomial is written with `f64::mul_add` on paths where the
//!   hardware has it, and with plain arithmetic where it does not (a software
//!   `fma()` call would be far slower than the multiply and add it replaces).
//! * **Estrin, not Horner.** Horner's 10-deep dependency chain cost more in
//!   latency than the entire rest of the kernel -- switching to a balanced tree
//!   took the portable path from 9.5 to 6.9 ns/element on its own.
//!
//! # Accuracy
//!
//! This does not trade accuracy for speed: it produces **bit-identical results
//! to the previous `(x as f64).tanh() as f32` on all 2^32 float32 inputs**, on
//! every dispatch path (AVX-512, AVX2+FMA, and portable). That was checked
//! exhaustively rather than sampled, by sweeping the entire float32 domain
//! through each block entry point; `tanh_matches_promoted_reference` below
//! re-checks a spread of ranges and every special value on each run. So the
//! accuracy argument recorded in `ops::activation::hyperbolic` -- worst
//! relative error 5.9e-08, ahead of NumPy's 1.1e-07 -- carries over unchanged.
//!
//! The polynomial degree was chosen against that same sweep rather than from
//! the error bound alone. Truncating it one term shorter (degree 11) still
//! matches on all 2^32 inputs; two terms shorter breaks 43 of them by one ulp.
//! Degree 12 is therefore the first degree with a whole term of margin, which
//! is what keeps the claim from resting on the exact `libm` the reference was
//! measured against. The cost of that margin is one FMA in about thirty.
//!
//! # Measured
//!
//! Single-threaded, 1M float32 elements, on a 4-core Xeon at 2.8GHz:
//!
//! ```text
//!     promoted scalar (previous)   21.4 ns/elem
//!     portable (this kernel)        6.8 ns/elem    3.1x
//!     AVX2 + FMA                    4.0 ns/elem    5.4x
//!     AVX-512                       2.0 ns/elem   10.4x
//! ```
//!
//! End to end from Python on the same machine, against NumPy's own SIMD
//! `tanh` -- this is the number that motivated the work:
//!
//! ```text
//!            N     before     after
//!         4096      31.3x      2.6x
//!        65536      16.7x      1.8x
//!      1048576      11.8x      1.1x
//!      4194304         --      0.6x   (faster than NumPy)
//! ```
//!
//! The residual gap at small sizes is structural, and is the accuracy
//! decision showing up as a cost: NumPy evaluates in float32 lanes (16 wide
//! under AVX-512) with a shorter polynomial, while this evaluates in float64
//! (8 wide) to keep the result exactly what the scalar promotion produced.
//!
//! # Not covered
//!
//! float64 `tanh` still calls `libm`. The same skeleton would serve it, but a
//! correctly-rounded float64 result needs the reduction and the polynomial
//! carried to ~2^-60 -- double-double residuals in places where the float32
//! path can round freely -- and that is a different piece of work. The float64
//! gap against NumPy is also much smaller to begin with (2.5x at a million
//! elements against float32's 11.8x).

use std::mem::MaybeUninit;

const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `ln 2` split in two, so the reduction can subtract it without rounding.
///
/// `LN2_HI` is `0x3FE6_2E42_FEE0_0000`: 21 zero bits at the bottom of the
/// mantissa, so `n * LN2_HI` is exact for any `|n| < 2^21` and in particular
/// for the `|n| <= 29` this kernel produces. `LN2_LO` carries the next 53 bits
/// of the true value, which puts `LN2_HI + LN2_LO` within a rounding of `ln 2`
/// -- far tighter than the polynomial that consumes it.
const LN2_HI: f64 = 0.6931471803691238;
const LN2_LO: f64 = 1.9082149292705877e-10;

/// Beyond this, `tanh` is 1.0 to float32 precision -- `1 - tanh(x) < 2^-25`
/// already at `x > 9.011` -- so the input is clamped and the formula runs on a
/// bounded range. Clamping rather than branching keeps the loop vectorizable,
/// and NaN fails both comparisons and falls through to a NaN result.
const LIMIT: f64 = 10.0;

/// `2^52 + 2^51`. Adding this to a value of magnitude `< 2^51` forces
/// round-to-nearest-even into the mantissa, so the low bits of the sum are the
/// rounded integer `n` and `sum - MAGIC` is `n` as a float.
const MAGIC: f64 = 6755399441055744.0;

#[inline(always)]
fn fma_or<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `exp(r) - 1` for `|r| <= ln2/2`, as `r + r^2 * Q(r)`.
///
/// Keeping the leading `r` outside the polynomial is what preserves relative
/// accuracy as `r -> 0`; folding it in would round it against terms it
/// dominates.
#[inline(always)]
fn expm1_poly<const FMA: bool>(r: f64) -> f64 {
    const C2: f64 = 1.0 / 2.0;
    const C3: f64 = 1.0 / 6.0;
    const C4: f64 = 1.0 / 24.0;
    const C5: f64 = 1.0 / 120.0;
    const C6: f64 = 1.0 / 720.0;
    const C7: f64 = 1.0 / 5040.0;
    const C8: f64 = 1.0 / 40320.0;
    const C9: f64 = 1.0 / 362880.0;
    const C10: f64 = 1.0 / 3628800.0;
    const C11: f64 = 1.0 / 39916800.0;
    const C12: f64 = 1.0 / 479001600.0;

    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let a = fma_or::<FMA>(r, C3, C2);
    let b = fma_or::<FMA>(r, C5, C4);
    let c = fma_or::<FMA>(r, C7, C6);
    let d = fma_or::<FMA>(r, C9, C8);
    let e = fma_or::<FMA>(r, C11, C10);
    let lo = fma_or::<FMA>(r2, b, a);
    let mid = fma_or::<FMA>(r2, d, c);
    let hi = fma_or::<FMA>(r2, C12, e);
    let q = fma_or::<FMA>(r8, hi, fma_or::<FMA>(r4, mid, lo));
    fma_or::<FMA>(r2, q, r)
}

/// One element. Branch-free by construction so the enclosing loop vectorizes.
#[inline(always)]
fn tanh_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    // `clamp` leaves NaN alone (both of its comparisons fail), which is what
    // carries a NaN input through to a NaN result.
    let t = xd.clamp(-LIMIT, LIMIT) * 2.0;

    // `z` carries the rounded `n` in its low bits and `2^n` in bits 52..63
    // after a shift; a NaN input makes both garbage, but `r` is then NaN too
    // and NaN wins every arithmetic step below.
    let z = fma_or::<FMA>(t, LOG2E, MAGIC);
    let n = z - MAGIC;
    let two_pow_n = f64::from_bits(z.to_bits().wrapping_add(1023) << 52);

    let r = fma_or::<FMA>(-n, LN2_LO, fma_or::<FMA>(-n, LN2_HI, t));
    let p = expm1_poly::<FMA>(r);
    let u = fma_or::<FMA>(two_pow_n, p, two_pow_n - 1.0);
    let y = (u / (u + 2.0)) as f32;

    // `tanh` is odd, so the result's sign is always the input's -- everywhere
    // except zero, where the arithmetic above loses it: at `x = -0.0` the
    // polynomial's `r * r` is `+0.0`, and `+0.0 + -0.0` is `+0.0` under
    // round-to-nearest, so `-0.0` comes back positive. Taking the sign from the
    // input restores it and is a no-op for every other value.
    y.copysign(x)
}

/// The loop LLVM actually vectorizes. Every instantiation below is this same
/// body compiled for a different instruction set.
#[inline(always)]
fn tanh_block<const FMA: bool>(input: &[f32], out: &mut [MaybeUninit<f32>]) {
    debug_assert_eq!(input.len(), out.len());
    for (o, &x) in out.iter_mut().zip(input.iter()) {
        o.write(tanh_one::<FMA>(x));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn tanh_block_avx512(input: &[f32], out: &mut [MaybeUninit<f32>]) {
    tanh_block::<true>(input, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn tanh_block_avx2(input: &[f32], out: &mut [MaybeUninit<f32>]) {
    tanh_block::<true>(input, out)
}

/// Which compilation of [`tanh_block`] this CPU gets. Resolved once per
/// operation by [`TanhF32Block::select`], not once per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    /// aarch64 has FMA and 128-bit vectors in its baseline, so the portable
    /// body already vectorizes there; it just needs `mul_add` turned on.
    #[cfg(target_arch = "aarch64")]
    NativeFma,
    /// No known-good hardware FMA: plain multiply and add, and whatever the
    /// baseline target offers for vector width.
    Portable,
}

/// A selected `tanh` kernel for float32.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TanhF32Block(Backend);

impl TanhF32Block {
    /// Pick the widest kernel the host supports.
    pub(crate) fn select() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let caps = crate::ops::simd::simd_capabilities();
            if caps.avx512 {
                return Self(Backend::Avx512);
            }
            // AVX2 without FMA exists (early VIA parts); `mul_add` would then
            // be a `libm` call, so check for it rather than assuming.
            if caps.avx2 && is_x86_feature_detected!("fma") {
                return Self(Backend::Avx2Fma);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self(Backend::NativeFma);
        }
        #[allow(unreachable_code)]
        Self(Backend::Portable)
    }

    /// Write `tanh(input[i])` into every element of `out`.
    ///
    /// On return every element of `out` has been initialized, which is what
    /// [`crate::ops::map::unary_map_blocks_threshold`] requires of it.
    #[inline]
    pub(crate) fn apply(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        match self.0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `select` returned this variant only after
            // `is_x86_feature_detected!` confirmed avx512f on this CPU.
            Backend::Avx512 => unsafe { tanh_block_avx512(input, out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above, for avx2 and fma.
            Backend::Avx2Fma => unsafe { tanh_block_avx2(input, out) },
            #[cfg(target_arch = "aarch64")]
            Backend::NativeFma => tanh_block::<true>(input, out),
            Backend::Portable => tanh_block::<false>(input, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(backend: Backend, xs: &[f32]) -> Vec<f32> {
        let mut out = vec![MaybeUninit::uninit(); xs.len()];
        TanhF32Block(backend).apply(xs, &mut out);
        out.into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect()
    }

    /// Every backend this CPU can actually run.
    fn available() -> Vec<Backend> {
        let mut v = vec![Backend::Portable];
        #[cfg(target_arch = "x86_64")]
        {
            let caps = crate::ops::simd::simd_capabilities();
            if caps.avx2 && is_x86_feature_detected!("fma") {
                v.push(Backend::Avx2Fma);
            }
            if caps.avx512 {
                v.push(Backend::Avx512);
            }
        }
        #[cfg(target_arch = "aarch64")]
        v.push(Backend::NativeFma);
        v
    }

    fn reference(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    /// This module's central claim, in runnable form: every float32 value maps
    /// to exactly the bits the scalar promotion produced, on every backend the
    /// host supports.
    ///
    /// Ignored by default because it walks all 2^32 of them, which takes about
    /// a minute on four cores. Run it with:
    ///
    /// ```text
    /// cargo test -p engine --release -- --ignored tanh_matches_promoted_reference_exhaustively
    /// ```
    #[test]
    #[ignore = "sweeps all 2^32 float32 inputs; takes ~1 minute"]
    fn tanh_matches_promoted_reference_exhaustively() {
        use rayon::prelude::*;

        const BLOCK: u64 = 8192;
        const BLOCKS: u64 = (1u64 << 32) / BLOCK;

        for backend in available() {
            let bad: u64 = (0..BLOCKS)
                .into_par_iter()
                .map(|b| {
                    let base = b * BLOCK;
                    let xs: Vec<f32> = (0..BLOCK)
                        .map(|k| f32::from_bits((base + k) as u32))
                        .collect();
                    let mut out = vec![MaybeUninit::uninit(); xs.len()];
                    TanhF32Block(backend).apply(&xs, &mut out);
                    xs.iter()
                        .zip(out)
                        .filter(|&(&x, ref o)| {
                            // SAFETY: `apply` initialized every element.
                            let got = unsafe { o.assume_init() };
                            let want = reference(x);
                            // NaN payloads are not part of the contract; every
                            // other value must match bit for bit.
                            got.to_bits() != want.to_bits() && !(got.is_nan() && want.is_nan())
                        })
                        .count() as u64
                })
                .sum();
            assert_eq!(bad, 0, "{backend:?}: {bad} of 2^32 inputs differ");
        }
    }

    /// The whole point of the kernel: same bits as the routine it replaces.
    /// The exhaustive 2^32 sweep lives outside the test suite (it takes ~50s);
    /// this covers each regime and every boundary the code branches on.
    #[test]
    fn tanh_matches_promoted_reference() {
        let mut xs: Vec<f32> = Vec::new();
        // Dense sweep across the interesting range, crossing every `n` step.
        for i in -200_000i32..200_000 {
            xs.push(i as f32 * 1e-4);
        }
        // Decades from subnormal to overflow-clamp and beyond.
        for e in -45i32..40 {
            let m = (2.0f64).powi(e) as f32;
            xs.extend_from_slice(&[m, -m, m * 1.5, -m * 1.5, m * 1.9999, -m * 1.9999]);
        }
        // The clamp boundary and the point where the result reaches 1.0.
        for &b in &[8.9f32, 9.0, 9.010913, 9.011, 9.1, 9.9, 10.0, 10.1, 20.0] {
            xs.extend_from_slice(&[b, -b]);
        }
        xs.extend_from_slice(&[
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);

        for backend in available() {
            for (&x, got) in xs.iter().zip(run(backend, &xs)) {
                assert_eq!(
                    got.to_bits(),
                    reference(x).to_bits(),
                    "{backend:?}: tanh({x:e}) gave {got:e}, want {:e}",
                    reference(x)
                );
            }
        }
    }

    #[test]
    fn tanh_propagates_nan() {
        for backend in available() {
            let out = run(backend, &[f32::NAN, -f32::NAN, 0.5]);
            assert!(out[0].is_nan(), "{backend:?}: NaN did not propagate");
            assert!(out[1].is_nan(), "{backend:?}: -NaN did not propagate");
            assert_eq!(out[2].to_bits(), reference(0.5).to_bits());
        }
    }

    /// Odd symmetry is a property callers rely on, and it is not automatic:
    /// the argument reduction rounds `n` independently for `x` and `-x`.
    #[test]
    fn tanh_is_odd() {
        for backend in available() {
            let xs: Vec<f32> = (1..5000).map(|i| i as f32 * 3e-3).collect();
            let neg: Vec<f32> = xs.iter().map(|v| -v).collect();
            for (p, n) in run(backend, &xs).into_iter().zip(run(backend, &neg)) {
                assert_eq!(p.to_bits(), (-n).to_bits(), "asymmetric at {p:e}");
            }
        }
    }

    /// Blocks are handed out by rayon at arbitrary lengths, so the tail past
    /// the last full vector has to be right too.
    #[test]
    fn tanh_handles_every_block_length() {
        let xs: Vec<f32> = (0..133).map(|i| (i as f32 - 66.0) * 0.21).collect();
        for backend in available() {
            for len in 0..xs.len() {
                let got = run(backend, &xs[..len]);
                for (&x, g) in xs[..len].iter().zip(got) {
                    assert_eq!(g.to_bits(), reference(x).to_bits(), "len {len}, x {x:e}");
                }
            }
        }
    }
}
