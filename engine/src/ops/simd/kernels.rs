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

/// Generates an element-wise unary block kernel for the rounding family —
/// `floor`, `ceil`, and round-to-nearest-even.
///
/// These need their own compilation for a different reason than
/// [`binary_elementwise!`]'s: not register width, but an instruction that
/// x86-64's baseline does not have at all. `roundps`/`roundpd` arrived with
/// SSE4.1, and until LLVM is allowed to use them there is no way to round a
/// vector, so it gives up on the loop entirely and emits one `floorf` call per
/// element. Measured over a million float32 elements on a 4-core container:
///
/// ```text
///   floor, baseline (SSE2)   2777.8 us
///   floor, SSE4.1             319.3 us   8.7x
/// ```
///
/// A 256-bit tier is not worth a third compilation here — `vroundps` measured
/// 339.3us, indistinguishable from SSE4.1, because at one instruction per
/// element the loop is bounded by memory rather than by issue width. aarch64
/// needs no tier at all: `frintm`/`frintp`/`frintn` are baseline NEON, so the
/// portable body already vectorizes there.
///
/// The one-parameter form exists for `round(decimals)`, which scales by a power
/// of ten before rounding and back afterwards; the parameter is captured once
/// per block rather than per element.
macro_rules! rounding_kernel {
    ($(#[$meta:meta])* $name:ident, $ty:ty, |$x:ident| $body:expr) => {
        rounding_kernel!($(#[$meta])* $name, $ty, |$x, _unused| $body);
    };
    ($(#[$meta:meta])* $name:ident, $ty:ty, |$x:ident, $p:ident| $body:expr) => {
        $(#[$meta])*
        pub(crate) fn $name(input: &[$ty], out: &mut [MaybeUninit<$ty>], param: $ty) {
            #[inline(always)]
            fn body(input: &[$ty], out: &mut [MaybeUninit<$ty>], $p: $ty) {
                // Reborrowing at the output length drops the bounds checks; see
                // `binary_elementwise!`.
                let n = out.len();
                let input = &input[..n];
                for i in 0..n {
                    let $x = input[i];
                    out[i].write($body);
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "sse4.1")]
            fn body_sse41(input: &[$ty], out: &mut [MaybeUninit<$ty>], p: $ty) {
                body(input, out, p)
            }

            #[cfg(target_arch = "x86_64")]
            if simd_capabilities().sse4_1 {
                // SAFETY: `detect` confirmed sse4.1 on this CPU.
                unsafe { body_sse41(input, out, param) };
                return;
            }

            body(input, out, param)
        }
    };
}

rounding_kernel!(
    /// Element-wise `floor` over a block of f32. `param` is ignored.
    floor_f32_blocks, f32, |x| x.floor()
);
rounding_kernel!(
    /// Element-wise `floor` over a block of f64. `param` is ignored.
    floor_f64_blocks, f64, |x| x.floor()
);
rounding_kernel!(
    /// Element-wise `ceil` over a block of f32. `param` is ignored.
    ceil_f32_blocks, f32, |x| x.ceil()
);
rounding_kernel!(
    /// Element-wise `ceil` over a block of f64. `param` is ignored.
    ceil_f64_blocks, f64, |x| x.ceil()
);
rounding_kernel!(
    /// Round a block of f32 to `param` decimal places, halves to even.
    round_f32_blocks, f32, |x, m| (x * m).round_ties_even() / m
);
rounding_kernel!(
    /// Round a block of f64 to `param` decimal places, halves to even.
    round_f64_blocks, f64, |x, m| (x * m).round_ties_even() / m
);

/// The float sum every reduction in the library bottoms out in: `LANES`
/// running chains, and a pairwise fold over 128-element leaves done in the
/// same pass.
///
/// The lanes are what let it vectorize -- one accumulator is a dependent chain
/// the compiler may not split, because floating point addition is not
/// associative. They also divide the error by the lane count, which was where
/// this stopped: eight lanes over 8192 elements is still a run of 1024
/// additions per lane, and the error of a run that deep grows with its length
/// where a pairwise fold's grows like `log n`. Averaged over 40 draws it was
/// 2.96 times NumPy's error at 8192 elements and 1.83 times at 1024, with a
/// worst case of 3.7e-7 against NumPy's 1.1e-7. Those are single-ulp figures --
/// a float32 sum is never far wrong -- but they were the shape that gets worse
/// with size rather than staying put, and this is the kernel every other
/// reduction is built on.
///
/// Folding 128-element leaves pairwise makes the depth logarithmic, which is
/// NumPy's algorithm and its leaf size; the mean error is now within noise of
/// NumPy's at every length. Two things about the shape of it, both measured:
///
/// * The obvious recursion -- `f(left) + f(right)` down to the leaf -- cost
///   2.4x. A call returning one float cannot keep its accumulators in registers
///   and pays for the slice setup again every 128 elements.
/// * Folding the lanes at each leaf, instead of once at the end, cost 25% on
///   `nansum`. The lanes stay apart the whole way here, so the inner loop is
///   still nothing but `LANES` independent adds, and a leaf boundary is
///   `LANES` more adds every 128 elements. That is a few percent of the
///   additions, and it measured 8-33% *faster* than the flat walk it replaced
///   -- shorter chains leave the out-of-order engine more to overlap.
///
/// `pending[k]` holds the lane totals of a run of `2^k` leaves that has not yet
/// found its partner, exactly as the bits of `leaves` say -- pushing a leaf
/// carries the ones upward like an increment, and each carry is one addition of
/// two equal-sized runs. That is the tree the recursion would have built,
/// without the calls: 40 slots is more leaves than a `usize` can index.
///
/// `keep` is applied to each element on the way in: the identity for `sum`, and
/// mapping NaN to zero for `nansum`, which is what lets the two agree
/// bit-for-bit on data without NaN in it.
macro_rules! float_sum_kernel {
    ($(#[$attr:meta])* $name:ident, $ty:ty, $lanes:expr, $keep:expr) => {
        $(#[$attr])*
        pub fn $name(data: &[$ty]) -> $ty {
            const LANES: usize = $lanes;
            /// NumPy's. Long enough that the merge below is noise against the
            /// adds, short enough that a lane's run through one leaf stays a
            /// handful of roundings.
            const LEAF: usize = 128;

            /// One leaf's worth of elements into `LANES` running chains.
            /// Nothing is folded here: the lanes stay apart all the way to the
            /// end, so the loop is `LANES` independent adds per iteration and
            /// nothing else -- which is what keeps this the same hot loop it
            /// was before the leaves existed.
            #[inline(always)]
            fn lane_sums(block: &[$ty], keep: impl Fn($ty) -> $ty) -> [$ty; LANES] {
                let mut sums = [0.0 as $ty; LANES];
                let mut chunks = block.chunks_exact(LANES);
                for chunk in &mut chunks {
                    for lane in 0..LANES {
                        sums[lane] += keep(chunk[lane]);
                    }
                }
                // A slice shorter than one vector still has to go somewhere,
                // and lane 0 is where a `LANES`-wide walk would have put it.
                for &v in chunks.remainder() {
                    sums[0] += keep(v);
                }
                sums
            }

            let keep = $keep;
            let mut pending = [[0.0 as $ty; LANES]; 40];
            let mut leaves: usize = 0;
            let mut blocks = data.chunks_exact(LEAF);

            for block in &mut blocks {
                // Carry: while a run of this size is already waiting, the two
                // combine and the result carries into the next size up. The
                // merge is `LANES` adds per leaf against the leaf's `LEAF`, so
                // it costs a few percent of the additions and nothing else.
                let mut sums = lane_sums(block, keep);
                let mut level = 0;
                while leaves & (1 << level) != 0 {
                    for lane in 0..LANES {
                        sums[lane] += pending[level][lane];
                    }
                    leaves &= !(1 << level);
                    level += 1;
                }
                pending[level] = sums;
                leaves |= 1 << level;
            }

            // What is left is one partial per set bit, each covering twice the
            // run of the bit below it. Combining them smallest-first pairs each
            // one with a partner its own size or larger, which is what keeps
            // the tree balanced; the tail, being the shortest run of all, is
            // where it starts. Which side of the `+` each lands on does not
            // matter -- addition is commutative even in floating point, and it
            // is the grouping that decides the rounding.
            let mut acc = lane_sums(blocks.remainder(), keep);
            let mut level = 0;
            let mut rest = leaves;
            while rest != 0 {
                if rest & 1 != 0 {
                    for lane in 0..LANES {
                        acc[lane] += pending[level][lane];
                    }
                }
                rest >>= 1;
                level += 1;
            }

            // Only now do the lanes meet, and pairwise: adding them up in order
            // would put `LANES` more roundings on the deepest value here.
            let mut width = LANES;
            while width > 1 {
                width /= 2;
                for lane in 0..width {
                    acc[lane] += acc[lane + width];
                }
            }
            acc[0]
        }
    };
}

float_sum_kernel!(
    /// Sum an f32 slice. See [`float_sum_kernel`].
    simd_sum_f32, f32, 8, |v| v
);

float_sum_kernel!(
    /// Sum an f64 slice. See [`float_sum_kernel`].
    simd_sum_f64, f64, 4, |v| v
);

float_sum_kernel!(
    /// Sum an f32 slice, skipping NaN. Branchless -- a NaN contributes the
    /// identity rather than taking a different path, so the loop still
    /// vectorizes and the result matches [`simd_sum_f32`] exactly when there is
    /// no NaN to skip.
    simd_nansum_f32, f32, 8, |v: f32| if v.is_nan() { 0.0 } else { v }
);

float_sum_kernel!(
    /// [`simd_nansum_f32`] in double precision.
    simd_nansum_f64, f64, 4, |v: f64| if v.is_nan() { 0.0 } else { v }
);

/// `simd_sum_f32` for the products of two slices, which is what a dot product
/// and every inner product built on one needs.
///
/// `dot` spelled this as `a.iter().zip(b).map(|(x, y)| x * y).sum()`, which is
/// a single dependent chain of multiply-adds: no vectorization, because
/// floating point addition cannot be reassociated without being told to, and
/// one rounding per element. It measured 7.5 times slower than NumPy's `sdot`
/// on 65536 elements and 36 times less accurate.
///
/// Only the common prefix is read, so a caller with mismatched lengths gets the
/// shorter one rather than a panic; `dot` checks the lengths itself.
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0f32; 8];
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut chunks = a.chunks_exact(8).zip(b.chunks_exact(8));
    for (x, y) in &mut chunks {
        sums[0] += x[0] * y[0];
        sums[1] += x[1] * y[1];
        sums[2] += x[2] * y[2];
        sums[3] += x[3] * y[3];
        sums[4] += x[4] * y[4];
        sums[5] += x[5] * y[5];
        sums[6] += x[6] * y[6];
        sums[7] += x[7] * y[7];
    }
    let mut total: f32 = sums.iter().sum();
    let tail = n - n % 8;
    for i in tail..n {
        total += a[i] * b[i];
    }
    total
}

/// [`simd_dot_f32`] in double precision; four lanes rather than eight, matching
/// [`simd_sum_f64`].
pub fn simd_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut sums = [0f64; 4];
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
    for (x, y) in &mut chunks {
        sums[0] += x[0] * y[0];
        sums[1] += x[1] * y[1];
        sums[2] += x[2] * y[2];
        sums[3] += x[3] * y[3];
    }
    let mut total: f64 = sums.iter().sum();
    let tail = n - n % 4;
    for i in tail..n {
        total += a[i] * b[i];
    }
    total
}

/// Unrolled sum for i32 slices to leverage auto-vectorization
pub fn simd_sum_i32(data: &[i32]) -> i32 {
    let mut sums = [0i32; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] = sums[0].wrapping_add(chunk[0]);
        sums[1] = sums[1].wrapping_add(chunk[1]);
        sums[2] = sums[2].wrapping_add(chunk[2]);
        sums[3] = sums[3].wrapping_add(chunk[3]);
        sums[4] = sums[4].wrapping_add(chunk[4]);
        sums[5] = sums[5].wrapping_add(chunk[5]);
        sums[6] = sums[6].wrapping_add(chunk[6]);
        sums[7] = sums[7].wrapping_add(chunk[7]);
    }
    let mut total: i32 = sums.iter().fold(0, |a, &b| a.wrapping_add(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_add(b));
    total
}

/// Sum an i32 slice into an i64 accumulator.
///
/// `sum` and `prod` report a wider integer than they read, matching NumPy and
/// PyTorch: a 32-bit total overflows after a few million counts and there is no
/// good answer to give once it has. Accumulating in i64 while *reading* i32 is
/// what keeps that from costing anything -- promoting the input first would
/// mean materializing a second, twice-as-large copy of it before the reduction
/// even starts.
///
/// `wrapping_add` on the i64 accumulator for the same reason the narrower
/// kernels use it: overflow past i64 stays two's-complement rather than
/// panicking in debug and wrapping in release. It takes 2^32 elements at full
/// magnitude to get there.
///
/// The widening is not free, and the cost is inherent rather than incidental:
/// every 256-bit load of eight i32 becomes two sign-extends and two 64-bit adds
/// where the same-width kernel did one. Summing 2M elements on four cores,
/// best-of-200 over four separate runs, went from ~0.055ms to ~0.122ms -- 2.2x
/// for the right answer. It is still around six times quicker than NumPy's
/// `int32` sum (~0.73ms), which widens the same way for the same reason.
///
/// Eight accumulators, not four or sixteen: those were tried and neither beat
/// this, sixteen clearly worse.
pub fn simd_sum_i32_to_i64(data: &[i32]) -> i64 {
    let mut sums = [0i64; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] = sums[0].wrapping_add(chunk[0] as i64);
        sums[1] = sums[1].wrapping_add(chunk[1] as i64);
        sums[2] = sums[2].wrapping_add(chunk[2] as i64);
        sums[3] = sums[3].wrapping_add(chunk[3] as i64);
        sums[4] = sums[4].wrapping_add(chunk[4] as i64);
        sums[5] = sums[5].wrapping_add(chunk[5] as i64);
        sums[6] = sums[6].wrapping_add(chunk[6] as i64);
        sums[7] = sums[7].wrapping_add(chunk[7] as i64);
    }
    let mut total: i64 = sums.iter().fold(0, |a, &b| a.wrapping_add(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_add(b as i64));
    total
}

/// Product of an i32 slice into an i64 accumulator. See
/// [`simd_sum_i32_to_i64`]; a product overflows far sooner than a sum, which is
/// exactly why the wider accumulator is worth having.
pub fn simd_prod_i32_to_i64(data: &[i32]) -> i64 {
    let mut prods = [1i64; 8];
    let chunks = data.chunks_exact(8);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] = prods[0].wrapping_mul(chunk[0] as i64);
        prods[1] = prods[1].wrapping_mul(chunk[1] as i64);
        prods[2] = prods[2].wrapping_mul(chunk[2] as i64);
        prods[3] = prods[3].wrapping_mul(chunk[3] as i64);
        prods[4] = prods[4].wrapping_mul(chunk[4] as i64);
        prods[5] = prods[5].wrapping_mul(chunk[5] as i64);
        prods[6] = prods[6].wrapping_mul(chunk[6] as i64);
        prods[7] = prods[7].wrapping_mul(chunk[7] as i64);
    }
    let mut total: i64 = prods.iter().fold(1, |a, &b| a.wrapping_mul(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_mul(b as i64));
    total
}

/// Unrolled sum for i64 slices to leverage auto-vectorization
pub fn simd_sum_i64(data: &[i64]) -> i64 {
    let mut sums = [0i64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        sums[0] = sums[0].wrapping_add(chunk[0]);
        sums[1] = sums[1].wrapping_add(chunk[1]);
        sums[2] = sums[2].wrapping_add(chunk[2]);
        sums[3] = sums[3].wrapping_add(chunk[3]);
    }
    let mut total: i64 = sums.iter().fold(0, |a, &b| a.wrapping_add(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_add(b));
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
        prods[0] = prods[0].wrapping_mul(chunk[0]);
        prods[1] = prods[1].wrapping_mul(chunk[1]);
        prods[2] = prods[2].wrapping_mul(chunk[2]);
        prods[3] = prods[3].wrapping_mul(chunk[3]);
        prods[4] = prods[4].wrapping_mul(chunk[4]);
        prods[5] = prods[5].wrapping_mul(chunk[5]);
        prods[6] = prods[6].wrapping_mul(chunk[6]);
        prods[7] = prods[7].wrapping_mul(chunk[7]);
    }
    let mut total: i32 = prods.iter().fold(1, |a, &b| a.wrapping_mul(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_mul(b));
    total
}

/// Unrolled product for i64 slices to leverage auto-vectorization
pub fn simd_prod_i64(data: &[i64]) -> i64 {
    let mut prods = [1i64; 4];
    let chunks = data.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        prods[0] = prods[0].wrapping_mul(chunk[0]);
        prods[1] = prods[1].wrapping_mul(chunk[1]);
        prods[2] = prods[2].wrapping_mul(chunk[2]);
        prods[3] = prods[3].wrapping_mul(chunk[3]);
    }
    let mut total: i64 = prods.iter().fold(1, |a, &b| a.wrapping_mul(b));
    total = rem.iter().fold(total, |a, &b| a.wrapping_mul(b));
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
