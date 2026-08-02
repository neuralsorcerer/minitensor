// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Vectorized float32 kernels for `tanh`, `erf`, both GELU variants, and the
//! two GELU gradients.
//!
//! These were the slowest things in the elementwise surface, all for the same
//! reason: a `libm` call per element, which no amount of rayon parallelism can
//! vectorize away. `tanh` measured 11.8x slower than NumPy's own SIMD `tanh` at
//! a million float32 elements and 31x at 4096, *after* spreading over four
//! cores; `erf` -- and so the exact GELU that every transformer uses -- ran at
//! 25 ns per element.
//!
//! The fix is to write each routine so that LLVM can vectorize it, then compile
//! it several times over -- once per instruction set -- and pick at runtime.
//! That requires the whole loop to live inside the multiversioned function, so
//! the entry points here take a block rather than an element (see
//! `ops::map::unary_map_blocks_threshold`, and its two-input sibling
//! `binary_map_blocks_threshold` for the gradients). All six share one
//! `select`, one argument reduction, and one set of dispatch machinery.
//!
//! Three details are what make them fast rather than merely correct:
//!
//! * **Rounding without `roundpd`.** `f64::round` is round-half-away-from-zero,
//!   which x86 cannot do in one instruction, so LLVM scalarizes the loop around
//!   it. Adding `MAGIC` instead forces the mantissa to hold the rounded
//!   integer, and the same bits then build `2^n` by a shift -- no float/int
//!   conversion either, which would otherwise scalarize again on Rust's
//!   saturating cast.
//! * **FMA, explicitly.** Rust never contracts `a*b + c` into an FMA on its
//!   own. The polynomials are written with `f64::mul_add` on paths where the
//!   hardware has it, and with plain arithmetic where it does not (a software
//!   `fma()` call would be far slower than the multiply and add it replaces).
//! * **Estrin, not Horner.** Horner's 10-deep dependency chain cost more in
//!   latency than the entire rest of the `tanh` kernel -- switching to a
//!   balanced tree took its portable path from 9.5 to 6.9 ns/element on its
//!   own. Every polynomial here is evaluated that way.
//!
//! # tanh
//!
//! `tanh(x) = u / (u + 2)` where `u = expm1(2x)`, in float64, rounded once.
//! Writing it through `expm1` rather than the more obvious `1 - 2/(exp(2x)+1)`
//! is what makes it accurate near zero: the latter cancels catastrophically as
//! `x -> 0`, while `u/(u+2)` has `u ~ 2x` over a denominator near 2 and stays
//! well conditioned.
//!
//! `expm1(t)` uses the textbook reduction `t = n*ln2 + r`, `|r| <= ln2/2`,
//! recombined as
//!
//! ```text
//!     expm1(t) = 2^n * p + (2^n - 1),   p = exp(r) - 1
//! ```
//!
//! which is the same expression for every `n` and cancels for none of them: at
//! `n = 0` it collapses to `p`, and for `n != 0` the `2^n - 1` term is bounded
//! away from zero exactly where `p` is small.
//!
//! # erf
//!
//! Two branches over `a = |x|`, selected branchlessly:
//!
//! ```text
//!     a <= 2   erf(a)  = a * (2/sqrt(pi) + t*G(t)),  t = a*a,  G degree 15
//!     a >  2   erfc(a) = exp(-t) * PB(1/a),                    PB degree 14
//! ```
//!
//! [`erf_parts`] returns both without combining them, because which
//! combination is safe depends on the caller -- see below. The split point and
//! both degrees were chosen by measuring the float32 rounding, not from an
//! error bound: 2 with degrees 15 and 14 was the cheapest pair that came out
//! correctly rounded, and the high branch is parameterized in `1/a` because `h`
//! is asymptotically `1/(a*sqrt(pi))`, which a polynomial in `1/a` captures in
//! far fewer terms than one in `a`.
//!
//! # GELU, and the cancellation it needs avoided
//!
//! Both variants are `x` times something that goes to zero as `x -> -inf`, and
//! the obvious spelling of that something destroys it:
//!
//! * `0.5 * x * (1 + erf(x/sqrt 2))`. For `x = -6.5`, `1 + erf` is 7.6e-11
//!   reconstructed from a float64 within an ulp of -1, so it keeps about five
//!   digits -- 24 ulp of error once `x` scales it back up. But `1 + erf(v)` is
//!   `erfc(-v)`, which the high branch of [`erf_parts`] already has in hand
//!   with no subtraction at all.
//! * `0.5 * x * (1 + tanh(v))` has the same problem and the same cure: it is
//!   `x * e/(e+1)` with `e = exp(2v)`, and `e` comes straight out of the
//!   reduction.
//!
//! Neither is a rounding nicety. The scalar float32 path this replaces returned
//! *exactly zero* from about `x = -5.5` down, for both variants.
//!
//! The gradients inherit both the problem and the cure. `Phi(x)` is the same
//! `1 + erf` in disguise, and `sech^2(v) = 1 - tanh(v)^2` collapses the same
//! way -- as `4*s*(1 - s)` with `s = e/(e+1)` it does not. This mattered for
//! float64 too, which is scalar and stayed so: its exact-GELU gradient was 1%
//! wrong at `x = -10` and had no correct digits past that, so both float64
//! spellings were fixed alongside.
//!
//! # Accuracy
//!
//! `tanh` is **bit-identical to the previous `(x as f64).tanh() as f32` on all
//! 2^32 float32 inputs**, on every dispatch path (AVX-512, AVX2+FMA, portable).
//! So the accuracy argument recorded in `ops::activation::hyperbolic` -- worst
//! relative error 5.9e-08, ahead of NumPy's 1.1e-07 -- carries over unchanged.
//! Its polynomial degree was picked against that same sweep: one term shorter
//! still matches everywhere, two terms shorter breaks 43 inputs, so degree 12
//! is the first with a whole term of margin.
//!
//! `erf` is within one ulp of the correctly rounded result everywhere, and is
//! *the* correctly rounded result on all but 68 of the 2^32 inputs. The
//! `libm::erff` it replaces misrounds 127,576,760 of them -- 2.97% -- so this
//! is a large accuracy gain as well as a 9x speedup.
//!
//! Both claims are checked exhaustively rather than sampled, by the ignored
//! tests at the bottom of this file; the ordinary tests re-check a spread of
//! ranges, every branch boundary and every special value on each run.
//!
//! # Measured
//!
//! Single-threaded, 1M float32 elements, on a 4-core Xeon at 2.8GHz:
//!
//! ```text
//!                        tanh    erf
//!     scalar (previous)  21.4   25.6  ns/elem
//!     portable            6.8   13.4
//!     AVX2 + FMA          4.0    6.2
//!     AVX-512             2.0    3.2
//! ```
//!
//! End to end from Python at a million elements, which is the number that
//! motivated the work:
//!
//! ```text
//!                        before    after
//!     tanh               7090us    606us   11.7x
//!     erf                6708us    848us    7.9x
//!     gelu (exact)       6471us    953us    6.8x
//!     gelu (tanh)        9340us    655us   14.3x
//!     gelu backward      7937us   2372us    3.3x
//! ```
//!
//! The backward figure is the whole gradient step, so about 1.3ms of it is
//! autograd graph and allocation overhead that this work does not touch -- the
//! kernel itself went from roughly 6.6ms to 1.1ms. It was the most expensive
//! gradient in the activation set, costing more than the forward pass.
//!
//! Against NumPy's SIMD `tanh`, float32 `tanh` goes from 11.8x slower at a
//! million elements to 1.1x, and is faster than NumPy past two million. The
//! residual gap at small sizes is the accuracy decision showing up as a cost:
//! NumPy evaluates in float32 lanes (16 wide under AVX-512) with a shorter
//! polynomial, while these evaluate in float64 (8 wide).
//!
//! # Not covered
//!
//! The float64 kernels all still call `libm`. The same skeleton would serve
//! them, but a correctly-rounded float64 result needs the reduction and the
//! polynomials carried to ~2^-60 -- double-double residuals in places where
//! the float32 path can round freely -- and that is a different piece of work.
//! The float64 gap is also much smaller to begin with (`tanh` is 2.5x NumPy at
//! a million elements against float32's 11.8x).
//!
//! `erfc` is untouched: it needs relative accuracy out where `erf` has
//! saturated, so it wants the high branch extended rather than reused.

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
const TANH_LIMIT: f64 = 10.0;

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

/// Split `v` into `2^n` and `r = v - n*ln2` with `|r| <= ln2/2`, the reduction
/// every exponential here starts from.
///
/// `z` carries the rounded `n` in its low bits, and the same bits shifted into
/// the exponent field give `2^n`. A NaN input makes both garbage, but `r` comes
/// back NaN, and NaN then wins every arithmetic step downstream.
#[inline(always)]
fn reduce_exp<const FMA: bool>(v: f64) -> (f64, f64) {
    let z = fma_or::<FMA>(v, LOG2E, MAGIC);
    let n = z - MAGIC;
    let two_pow_n = f64::from_bits(z.to_bits().wrapping_add(1023) << 52);
    let r = fma_or::<FMA>(-n, LN2_LO, fma_or::<FMA>(-n, LN2_HI, v));
    (two_pow_n, r)
}

/// `tanh` in float64. Branch-free by construction so callers vectorize.
#[inline(always)]
fn tanh_core<const FMA: bool>(xd: f64) -> f64 {
    // `clamp` leaves NaN alone (both of its comparisons fail), which is what
    // carries a NaN input through to a NaN result.
    let t = xd.clamp(-TANH_LIMIT, TANH_LIMIT) * 2.0;
    let (two_pow_n, r) = reduce_exp::<FMA>(t);
    let p = expm1_poly::<FMA>(r);
    let u = fma_or::<FMA>(two_pow_n, p, two_pow_n - 1.0);
    let y = u / (u + 2.0);

    // `tanh` is odd, so the result's sign is always the input's -- everywhere
    // except zero, where the arithmetic above loses it: at `x = -0.0` the
    // polynomial's `r * r` is `+0.0`, and `+0.0 + -0.0` is `+0.0` under
    // round-to-nearest, so `-0.0` comes back positive. Taking the sign from the
    // input restores it and is a no-op for every other value.
    y.copysign(xd)
}

#[inline(always)]
fn tanh_one<const FMA: bool>(x: f32) -> f32 {
    tanh_core::<FMA>(x as f64) as f32
}

// ---------------------------------------------------------------------------
// erf
// ---------------------------------------------------------------------------

/// `erf` reaches 1.0 in *float32* well before here (`1 - erf(x) < 2^-25`
/// already at `x > 3.92`), but the clamp sits at 6 rather than 4 because GELU
/// needs more: it forms `1 + erf(x/sqrt 2)` and multiplies by `x`, so a
/// clamped `erf` of 0.99999998 leaks a residual that `x` then amplifies --
/// `gelu(-20)` came back as -1.5e-7 instead of -0. GELU's negative tail is
/// `0.5*x*erfc(|x|/sqrt 2)`, which stays a representable float32 down to about
/// `x = -14`; 11 is where that argument lands, so the tail is computed rather
/// than clamped for every `x` whose result is not already zero.
/// As with `tanh`, clamping keeps the loop branch-free and lets NaN through.
const ERF_LIMIT: f64 = 11.0;

/// Where the polynomial branch hands over to the `exp`-based one. Chosen by
/// measurement -- see the module docs.
const ERF_SPLIT: f64 = 2.0;

/// `erf(x) = |x| * (A00 + t*G(t))`, `t = x*x`, for `|x| <= 2`.
///
/// `A00` is `2/sqrt(pi)` correctly rounded and deliberately sits outside the
/// polynomial: that makes the small-`x` limit `erf(x) -> 2x/sqrt(pi)` exact.
/// Folding it in cost 1 ulp on inputs down around `1e-6`, which is how it was
/// found.
const A00: f64 = std::f64::consts::FRAC_2_SQRT_PI;
const G00: f64 = -0.3761263890318367;
const G01: f64 = 0.11283791670949633;
const G02: f64 = -0.026866170644324023;
const G03: f64 = 0.005223977619979951;
const G04: f64 = -0.0008548326818884021;
const G05: f64 = 0.00012055328255711209;
const G06: f64 = -1.4925578101206095e-05;
const G07: f64 = 1.646134799302676e-06;
const G08: f64 = -1.6360026129096463e-07;
const G09: f64 = 1.4774934544966013e-08;
const G10: f64 = -1.2158547830560174e-09;
const G11: f64 = 9.022812758721981e-11;
const G12: f64 = -5.817801680396542e-12;
const G13: f64 = 3.005698198104101e-13;
const G14: f64 = -1.0701644628115967e-14;
const G15: f64 = 1.8933956626349782e-16;

/// `erf(x) = 1 - exp(-t) * PB(1/|x|)` for `2 < |x| <= 11`, where
/// `PB(a) ~ erfc(a)*exp(a^2)`, degree 14 in `1/a`.
///
/// In `1/a` rather than `a` because `h` is asymptotically `1/(a*sqrt(pi))`:
/// over [2,11] a degree-14 fit in `1/a` holds 2^-40 where a fit in `a` falls
/// apart. It costs one division, measured at 0.26 ns per element -- cheaper
/// than the extra terms, and it is what lets the range widen this far without
/// losing accuracy on (2,4], which is the only part `erf` itself needs.
const B00: f64 = 1.8179156235920146e-09;
const B01: f64 = 0.5641894676558621;
const B02: f64 = 3.1547652652352993e-06;
const B03: f64 = -0.28214063825216956;
const B04: f64 = 0.0003249173235522797;
const B05: f64 = 0.4235999491180208;
const B06: f64 = -0.03573853362794864;
const B07: f64 = -0.6468243562176176;
const B08: f64 = -2.770024822999806;
const B09: f64 = 16.152979524845946;
const B10: f64 = -37.46625850802945;
const B11: f64 = 51.70220402502904;
const B12: f64 = -44.66697822274648;
const B13: f64 = 22.563581018752625;
const B14: f64 = -5.130091355864232;

/// `exp(r) - 1 = r * (E1 + r*(E2 + ...))` for `|r| <= ln2/2`.
///
/// Only degree 7, far shorter than [`expm1_poly`], because this one is used
/// solely inside erf's high branch where its error reaches the result damped
/// by `erfc/erf <= 0.0047`. Degree 6 already suffices; 7 is the usual spare
/// term.
const E1: f64 = 1.0;
const E2: f64 = 0.5;
const E3: f64 = 0.16666666666666666;
const E4: f64 = 0.041666666666666664;
const E5: f64 = 0.008333333333333333;
const E6: f64 = 0.001388888888888889;
const E7: f64 = 0.0001984126984126984;

/// The two branch values `erf` and `erfc` are assembled from, for the clamped
/// `a = |x|`: `erf(a)` on `[0,2]` and `erfc(a)` on `(2,6]`.
///
/// Returned separately rather than combined because the caller decides which
/// combination is safe. `erf` wants `1 - erfc`; GELU wants the `erfc` itself,
/// and reconstructing that as `1 - erf` would cancel away its digits -- at
/// `x = -6.5` that cost 24 ulp before the split.
#[inline(always)]
fn erf_parts<const FMA: bool>(a: f64) -> (f64, f64) {
    let t = a * a;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;

    // Low branch: |x| <= 2. Estrin over the degree-15 tail.
    let c0 = fma_or::<FMA>(t, G01, G00);
    let c1 = fma_or::<FMA>(t, G03, G02);
    let c2 = fma_or::<FMA>(t, G05, G04);
    let c3 = fma_or::<FMA>(t, G07, G06);
    let c4 = fma_or::<FMA>(t, G09, G08);
    let c5 = fma_or::<FMA>(t, G11, G10);
    let c6 = fma_or::<FMA>(t, G13, G12);
    let c7 = fma_or::<FMA>(t, G15, G14);
    let d0 = fma_or::<FMA>(t2, c1, c0);
    let d1 = fma_or::<FMA>(t2, c3, c2);
    let d2 = fma_or::<FMA>(t2, c5, c4);
    let d3 = fma_or::<FMA>(t2, c7, c6);
    let g = fma_or::<FMA>(t8, fma_or::<FMA>(t4, d3, d2), fma_or::<FMA>(t4, d1, d0));
    let low = a * fma_or::<FMA>(t, g, A00);

    // High branch: 2 < |x| <= 6, Estrin in `u = 1/a`. For low-branch inputs
    // `u` can be infinite (at x = 0) and this whole branch garbage, which is
    // fine: the select below discards it, and IEEE does not trap.
    let u = 1.0 / a;
    let u2 = u * u;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let e0 = fma_or::<FMA>(u, B01, B00);
    let e1 = fma_or::<FMA>(u, B03, B02);
    let e2 = fma_or::<FMA>(u, B05, B04);
    let e3 = fma_or::<FMA>(u, B07, B06);
    let e4 = fma_or::<FMA>(u, B09, B08);
    let e5 = fma_or::<FMA>(u, B11, B10);
    let e6 = fma_or::<FMA>(u, B13, B12);
    let f0 = fma_or::<FMA>(u2, e1, e0);
    let f1 = fma_or::<FMA>(u2, e3, e2);
    let f2 = fma_or::<FMA>(u2, e5, e4);
    let f3 = fma_or::<FMA>(u2, B14, e6);
    let pb = fma_or::<FMA>(u8, fma_or::<FMA>(u4, f3, f2), fma_or::<FMA>(u4, f1, f0));

    let (two_pow_n, r) = reduce_exp::<FMA>(-t);
    let mut s = fma_or::<FMA>(r, E7, E6);
    s = fma_or::<FMA>(r, s, E5);
    s = fma_or::<FMA>(r, s, E4);
    s = fma_or::<FMA>(r, s, E3);
    s = fma_or::<FMA>(r, s, E2);
    s = fma_or::<FMA>(r, s, E1);
    let exp_neg_t = two_pow_n * fma_or::<FMA>(r, s, 1.0);
    // `high` is erfc(a), not erf(a): the subtraction from 1 is left to the
    // caller so that callers who want the small quantity never do it.
    let high = exp_neg_t * pb;

    (low, high)
}

/// `erf` in float64, accurate enough that rounding to float32 is the correctly
/// rounded result on all but 68 of the 2^32 inputs.
#[inline(always)]
fn erf_core<const FMA: bool>(xd: f64) -> f64 {
    let a = xd.abs().clamp(0.0, ERF_LIMIT);
    let (erf_low, erfc_high) = erf_parts::<FMA>(a);
    // NaN fails `a <= ERF_SPLIT` and takes the high branch, which is NaN there.
    let y = if a <= ERF_SPLIT {
        erf_low
    } else {
        1.0 - erfc_high
    };
    // Odd, and the same signed-zero argument as `tanh_core`.
    y.copysign(xd)
}

#[inline(always)]
fn erf_one<const FMA: bool>(x: f32) -> f32 {
    erf_core::<FMA>(x as f64) as f32
}

/// `1 + erf(v)`, kept accurate on the `v < 0` tail where it goes to zero.
///
/// `1 + erf(v)` is `erfc(-v)`, so the negative side reads the `erfc` branch
/// straight out of [`erf_parts`] instead of adding 1 to something within an
/// ulp of -1. The positive side is `2 - erfc(v)`, which never cancels.
#[inline(always)]
fn one_plus_erf<const FMA: bool>(v: f64) -> f64 {
    let av = v.abs();
    let a = av.clamp(0.0, ERF_LIMIT);
    let (erf_low, erfc_high) = erf_parts::<FMA>(a);
    let erfc_a = if a <= ERF_SPLIT {
        1.0 - erf_low
    } else {
        erfc_high
    };
    // Past the fitted range, snap to zero rather than keep the clamped value.
    // The caller scales by `0.5 * x` and `x` is `sqrt(2) * v`, so `x` grows
    // linearly while `erfc(v)` decays like `exp(-v^2)`: once the product has
    // underflowed float32 it stays underflowed, and the clamped `erfc(11)`
    // would be enormously too large. NaN fails this comparison and survives.
    let erfc_a = if av > ERF_LIMIT { 0.0 } else { erfc_a };
    if v < 0.0 { erfc_a } else { 2.0 - erfc_a }
}

/// Exact GELU: `0.5 * x * (1 + erf(x/sqrt 2))`, in float64 throughout.
#[inline(always)]
fn gelu_erf_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let cdf = one_plus_erf::<FMA>(xd * std::f64::consts::FRAC_1_SQRT_2);
    (0.5 * xd * cdf) as f32
}

/// Clamp on the tanh-GELU inner argument. It only has to be wide enough that
/// `1 + tanh(v)` has already underflowed the float32 result, which happens far
/// inside the range where `2^n` still fits an exponent field: `v = -100` gives
/// `1 + tanh(v) ~ 1e-87`, and the `x` that produces it is about -14.
const GELU_TANH_LIMIT: f64 = 100.0;

/// Tanh-approximation GELU:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
///
/// Written as `x * e/(e+1)` with `e = exp(2v)`, which is the same function
/// with the cancellation taken out: `1 + tanh(v)` collapses to zero for
/// negative `v`, and `e` comes straight from the reduction with its digits
/// intact. The old scalar path computed `1 + tanhf(v)` in float32 and returned
/// exactly 0 from about `x = -5.5` down.
#[inline(always)]
fn gelu_tanh_one<const FMA: bool>(x: f32) -> f32 {
    const COEFF: f64 = 0.7978845608028654; // sqrt(2/pi)
    const CUBIC: f64 = 0.044715;
    let xd = x as f64;
    let inner = COEFF * fma_or::<FMA>(CUBIC * xd, xd * xd, xd);
    let (two_pow_n, r) = reduce_exp::<FMA>(inner.clamp(-GELU_TANH_LIMIT, GELU_TANH_LIMIT) * 2.0);
    let e = two_pow_n * (1.0 + expm1_poly::<FMA>(r));
    // Below the clamp the factor has already underflowed the float32 result,
    // so make it exactly zero rather than a clamped floor that an arbitrarily
    // large `x` could scale back up. This also keeps `x = -inf` at NaN, which
    // is what both the scalar path this replaces and the exact GELU return.
    // NaN fails the comparison and carries through the other way.
    let factor = if inner < -GELU_TANH_LIMIT {
        0.0
    } else {
        e / (e + 1.0)
    };
    (xd * factor) as f32
}

// ---------------------------------------------------------------------------
// GELU backward
// ---------------------------------------------------------------------------

/// Clamp on `x^2/2` inside the exact-GELU derivative. `exp(-300)` is 5e-131,
/// and the `x` that multiplies it is about 24, so the `x*pdf` term has long
/// since underflowed the float32 result; the clamp only keeps the exponent
/// field in range for absurd inputs.
const GELU_PDF_LIMIT: f64 = 300.0;

/// `1/sqrt(2*pi)`, the normal density's normalization.
const INV_SQRT_2PI: f64 = 0.3989422804014327;

/// `d/dx [x * Phi(x)] = Phi(x) + x * phi(x)`, times the incoming gradient.
#[inline(always)]
fn gelu_erf_backward_one<const FMA: bool>(x: f32, gout: f32) -> f32 {
    let xd = x as f64;
    // `one_plus_erf` is already the cancellation-free form, so `cdf` keeps its
    // digits into the negative tail where it decays to zero.
    let cdf = 0.5 * one_plus_erf::<FMA>(xd * std::f64::consts::FRAC_1_SQRT_2);
    let (two_pow_n, r) = reduce_exp::<FMA>(-(0.5 * xd * xd).min(GELU_PDF_LIMIT));
    let pdf = two_pow_n * (1.0 + expm1_poly::<FMA>(r)) * INV_SQRT_2PI;
    (fma_or::<FMA>(xd, pdf, cdf) * gout as f64) as f32
}

/// Derivative of the tanh-approximation GELU, times the incoming gradient.
///
/// Written through `e = exp(2v)` for the same reason the forward pass is:
/// `1 + tanh(v)` and `sech^2(v) = 1 - tanh(v)^2` both collapse to zero for
/// negative `v`, and subtracting from 1 destroys them. As products of
/// `e/(e+1)` they keep their digits -- `sech^2` is `4e/(e+1)^2`.
#[inline(always)]
fn gelu_tanh_backward_one<const FMA: bool>(x: f32, gout: f32) -> f32 {
    const COEFF: f64 = 0.7978845608028654; // sqrt(2/pi)
    const CUBIC: f64 = 0.044715;
    let xd = x as f64;
    let x2 = xd * xd;
    let inner = COEFF * fma_or::<FMA>(CUBIC * xd, x2, xd);
    let (two_pow_n, r) = reduce_exp::<FMA>(inner.clamp(-GELU_TANH_LIMIT, GELU_TANH_LIMIT) * 2.0);
    let e = two_pow_n * (1.0 + expm1_poly::<FMA>(r));
    let recip = 1.0 / (e + 1.0);
    let saturated = inner < -GELU_TANH_LIMIT;
    // 0.5*(1 + tanh) = e/(e+1);  sech^2 = 4e/(e+1)^2.
    let half_one_plus_tanh = if saturated { 0.0 } else { e * recip };
    let sech2 = if saturated {
        0.0
    } else {
        4.0 * e * recip * recip
    };
    let d_inner = COEFF * fma_or::<FMA>(3.0 * CUBIC, x2, 1.0);
    let local = fma_or::<FMA>(0.5 * xd * sech2, d_inner, half_one_plus_tanh);
    (local * gout as f64) as f32
}

/// Generate the block loop LLVM vectorizes, plus one compilation of it per
/// instruction set. Every kernel in this module is the same shape: a
/// branch-free element function, wrapped in a loop, compiled several times.
macro_rules! block_kernel {
    ($block:ident, $one:ident, $avx512:ident, $avx2:ident) => {
        #[inline(always)]
        fn $block<const FMA: bool>(input: &[f32], out: &mut [MaybeUninit<f32>]) {
            debug_assert_eq!(input.len(), out.len());
            for (o, &x) in out.iter_mut().zip(input.iter()) {
                o.write($one::<FMA>(x));
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        fn $avx512(input: &[f32], out: &mut [MaybeUninit<f32>]) {
            $block::<true>(input, out)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        fn $avx2(input: &[f32], out: &mut [MaybeUninit<f32>]) {
            $block::<true>(input, out)
        }
    };
}

/// The two-input form, for gradient kernels: saved input and incoming
/// gradient in, gradient out.
macro_rules! block_kernel2 {
    ($block:ident, $one:ident, $avx512:ident, $avx2:ident) => {
        #[inline(always)]
        fn $block<const FMA: bool>(lhs: &[f32], rhs: &[f32], out: &mut [MaybeUninit<f32>]) {
            debug_assert_eq!(lhs.len(), out.len());
            debug_assert_eq!(rhs.len(), out.len());
            for ((o, &x), &g) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
                o.write($one::<FMA>(x, g));
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        fn $avx512(lhs: &[f32], rhs: &[f32], out: &mut [MaybeUninit<f32>]) {
            $block::<true>(lhs, rhs, out)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        fn $avx2(lhs: &[f32], rhs: &[f32], out: &mut [MaybeUninit<f32>]) {
            $block::<true>(lhs, rhs, out)
        }
    };
}

/// Dispatch for [`block_kernel2!`].
macro_rules! dispatch2 {
    ($self:expr, $lhs:expr, $rhs:expr, $out:expr, $block:ident, $avx512:ident, $avx2:ident) => {
        match $self.0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `select` returned this variant only after
            // `is_x86_feature_detected!` confirmed avx512f on this CPU.
            Backend::Avx512 => unsafe { $avx512($lhs, $rhs, $out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above, for avx2 and fma.
            Backend::Avx2Fma => unsafe { $avx2($lhs, $rhs, $out) },
            #[cfg(target_arch = "aarch64")]
            Backend::NativeFma => $block::<true>($lhs, $rhs, $out),
            Backend::Portable => $block::<false>($lhs, $rhs, $out),
        }
    };
}

block_kernel!(tanh_block, tanh_one, tanh_block_avx512, tanh_block_avx2);
block_kernel!(erf_block, erf_one, erf_block_avx512, erf_block_avx2);
block_kernel!(
    gelu_erf_block,
    gelu_erf_one,
    gelu_erf_block_avx512,
    gelu_erf_block_avx2
);
block_kernel!(
    gelu_tanh_block,
    gelu_tanh_one,
    gelu_tanh_block_avx512,
    gelu_tanh_block_avx2
);
block_kernel2!(
    gelu_erf_backward_block,
    gelu_erf_backward_one,
    gelu_erf_backward_block_avx512,
    gelu_erf_backward_block_avx2
);
block_kernel2!(
    gelu_tanh_backward_block,
    gelu_tanh_backward_one,
    gelu_tanh_backward_block_avx512,
    gelu_tanh_backward_block_avx2
);

/// Dispatch one selected backend to the right compilation of a kernel.
macro_rules! dispatch {
    ($self:expr, $input:expr, $out:expr, $block:ident, $avx512:ident, $avx2:ident) => {
        match $self.0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `select` returned this variant only after
            // `is_x86_feature_detected!` confirmed avx512f on this CPU.
            Backend::Avx512 => unsafe { $avx512($input, $out) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above, for avx2 and fma.
            Backend::Avx2Fma => unsafe { $avx2($input, $out) },
            #[cfg(target_arch = "aarch64")]
            Backend::NativeFma => $block::<true>($input, $out),
            Backend::Portable => $block::<false>($input, $out),
        }
    };
}

/// Which compilation of the block kernels this CPU gets. Resolved once per
/// operation by [`F32Kernel::select`], not once per element.
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

/// A selected set of float32 kernels. One `select` covers every operation
/// here, so a caller that needs two of them pays for detection once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct F32Kernel(Backend);

impl F32Kernel {
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
    /// [`crate::ops::map::unary_map_blocks_threshold`] requires of it. The
    /// same holds for the other three.
    #[inline]
    pub(crate) fn tanh(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            tanh_block,
            tanh_block_avx512,
            tanh_block_avx2
        )
    }

    /// Write `erf(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn erf(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            erf_block,
            erf_block_avx512,
            erf_block_avx2
        )
    }

    /// Write the exact GELU of every element of `input` into `out`.
    #[inline]
    pub(crate) fn gelu_erf(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            gelu_erf_block,
            gelu_erf_block_avx512,
            gelu_erf_block_avx2
        )
    }

    /// Write the tanh-approximation GELU of every element into `out`.
    #[inline]
    pub(crate) fn gelu_tanh(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            gelu_tanh_block,
            gelu_tanh_block_avx512,
            gelu_tanh_block_avx2
        )
    }

    /// Write the exact-GELU gradient for `input`, scaled by `grad`, into `out`.
    #[inline]
    pub(crate) fn gelu_erf_backward(
        self,
        input: &[f32],
        grad: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        dispatch2!(
            self,
            input,
            grad,
            out,
            gelu_erf_backward_block,
            gelu_erf_backward_block_avx512,
            gelu_erf_backward_block_avx2
        )
    }

    /// Write the tanh-approximation GELU gradient into `out`.
    #[inline]
    pub(crate) fn gelu_tanh_backward(
        self,
        input: &[f32],
        grad: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        dispatch2!(
            self,
            input,
            grad,
            out,
            gelu_tanh_backward_block,
            gelu_tanh_backward_block_avx512,
            gelu_tanh_backward_block_avx2
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which kernel a test is exercising. Every test runs against every backend
    /// the host supports, so a vectorized compilation that disagrees with the
    /// portable one fails rather than passing quietly.
    #[derive(Clone, Copy, Debug)]
    enum Op {
        Tanh,
        Erf,
        GeluErf,
        GeluTanh,
    }

    fn apply(op: Op, backend: Backend, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        let k = F32Kernel(backend);
        match op {
            Op::Tanh => k.tanh(input, out),
            Op::Erf => k.erf(input, out),
            Op::GeluErf => k.gelu_erf(input, out),
            Op::GeluTanh => k.gelu_tanh(input, out),
        }
    }

    fn run(op: Op, backend: Backend, xs: &[f32]) -> Vec<f32> {
        let mut out = vec![MaybeUninit::uninit(); xs.len()];
        apply(op, backend, xs, &mut out);
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

    /// The float64 routine each kernel is measured against, rounded once.
    fn reference(op: Op, x: f32) -> f32 {
        let xd = x as f64;
        match op {
            Op::Tanh => xd.tanh() as f32,
            Op::Erf => libm::erf(xd) as f32,
            // Both GELU references are written cancellation-free, matching
            // the kernels. `1 + erf(u)` is `erfc(-u)`, and
            // `0.5*(1 + tanh(v))` is the logistic `1/(1 + exp(-2v))`. Spelling
            // them the naive way would make the *reference* the inaccurate
            // side of the comparison in the negative tail.
            Op::GeluErf => (0.5 * xd * libm::erfc(-xd * std::f64::consts::FRAC_1_SQRT_2)) as f32,
            Op::GeluTanh => {
                let inner = 0.7978845608028654 * (xd + 0.044715 * xd * xd * xd);
                (xd / (1.0 + (-2.0 * inner).exp())) as f32
            }
        }
    }

    /// Signed distance in representable float32 steps.
    fn ulps_apart(a: f32, b: f32) -> i64 {
        let key = |v: f32| -> i64 {
            let bits = v.to_bits() as i32;
            if bits < 0 {
                (i32::MIN as i64) - (bits as i64)
            } else {
                bits as i64
            }
        };
        (key(a) - key(b)).abs()
    }

    /// A spread of inputs covering every regime these kernels branch on.
    fn sample_inputs() -> Vec<f32> {
        let mut xs: Vec<f32> = Vec::new();
        for i in -200_000i32..200_000 {
            xs.push(i as f32 * 1e-4);
        }
        for e in -45i32..40 {
            let m = (2.0f64).powi(e) as f32;
            xs.extend_from_slice(&[m, -m, m * 1.5, -m * 1.5, m * 1.9999, -m * 1.9999]);
        }
        // The clamp and split boundaries of both kernels.
        for &b in &[
            1.9f32, 1.9999, 2.0, 2.0001, 2.1, 3.9, 3.99, 4.0, 4.01, 8.9, 9.010913, 9.011, 10.0,
            10.1, 20.0,
        ] {
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
        xs
    }

    /// `tanh`'s contract is the strong one: the same bits as the scalar
    /// promotion it replaced. The exhaustive sweep lives below; this covers
    /// each regime and every boundary on an ordinary test run.
    #[test]
    fn tanh_matches_promoted_reference() {
        let xs = sample_inputs();
        for backend in available() {
            for (&x, got) in xs.iter().zip(run(Op::Tanh, backend, &xs)) {
                assert_eq!(
                    got.to_bits(),
                    reference(Op::Tanh, x).to_bits(),
                    "{backend:?}: tanh({x:e})"
                );
            }
        }
    }

    /// `erf` and the two GELUs are not bit-exact against float64 -- they are
    /// within one ulp of it, which is still far better than the scalar `erff`
    /// and `tanhf` they replaced (2.97% of all float32 inputs misrounded).
    #[test]
    fn erf_and_gelu_stay_within_one_ulp() {
        let xs = sample_inputs();
        for op in [Op::Erf, Op::GeluErf, Op::GeluTanh] {
            for backend in available() {
                for (&x, got) in xs.iter().zip(run(op, backend, &xs)) {
                    let want = reference(op, x);
                    if want.is_nan() {
                        assert!(got.is_nan(), "{op:?}/{backend:?}: {x:e} -> {got:e}");
                        continue;
                    }
                    assert!(
                        ulps_apart(got, want) <= 1,
                        "{op:?}/{backend:?}: f({x:e}) gave {got:e}, want {want:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn propagates_nan() {
        for op in [Op::Tanh, Op::Erf, Op::GeluErf, Op::GeluTanh] {
            for backend in available() {
                let out = run(op, backend, &[f32::NAN, -f32::NAN, 0.5]);
                assert!(out[0].is_nan(), "{op:?}/{backend:?}: NaN");
                assert!(out[1].is_nan(), "{op:?}/{backend:?}: -NaN");
                assert!(out[2].is_finite());
            }
        }
    }

    /// Odd symmetry is not automatic: the argument reduction rounds `n`
    /// independently for `x` and `-x`, and erf picks its branch from `|x|`.
    #[test]
    fn odd_functions_stay_odd() {
        let xs: Vec<f32> = (1..5000).map(|i| i as f32 * 1e-3).collect();
        let neg: Vec<f32> = xs.iter().map(|v| -v).collect();
        for op in [Op::Tanh, Op::Erf, Op::GeluErf, Op::GeluTanh] {
            // GELU is not odd, but x*Phi(x) has x*Phi(-x) as its mirror; only
            // the genuinely odd kernels are checked for exact antisymmetry.
            if matches!(op, Op::GeluErf | Op::GeluTanh) {
                continue;
            }
            for backend in available() {
                for (p, n) in run(op, backend, &xs)
                    .into_iter()
                    .zip(run(op, backend, &neg))
                {
                    assert_eq!(p.to_bits(), (-n).to_bits(), "{op:?}/{backend:?}: {p:e}");
                }
            }
        }
    }

    /// Signed zero survives, which the `copysign` at the end of each core is
    /// there for -- the arithmetic loses it (`+0.0 + -0.0` is `+0.0`).
    #[test]
    fn signed_zero_survives() {
        for op in [Op::Tanh, Op::Erf, Op::GeluErf, Op::GeluTanh] {
            for backend in available() {
                let out = run(op, backend, &[0.0, -0.0]);
                assert!(
                    out[0] == 0.0 && !out[0].is_sign_negative(),
                    "{op:?}/{backend:?}: +0.0 -> {:e}",
                    out[0]
                );
                assert!(
                    out[1] == 0.0 && out[1].is_sign_negative(),
                    "{op:?}/{backend:?}: -0.0 -> {:e}",
                    out[1]
                );
            }
        }
    }

    /// Blocks arrive from rayon at arbitrary lengths, so the tail past the last
    /// full vector has to be right too.
    #[test]
    fn handles_every_block_length() {
        let xs: Vec<f32> = (0..133).map(|i| (i as f32 - 66.0) * 0.11).collect();
        for op in [Op::Tanh, Op::Erf, Op::GeluErf, Op::GeluTanh] {
            for backend in available() {
                for len in 0..xs.len() {
                    for (&x, g) in xs[..len].iter().zip(run(op, backend, &xs[..len])) {
                        assert!(
                            ulps_apart(g, reference(op, x)) <= 1,
                            "{op:?}/{backend:?}: len {len}, x {x:e}"
                        );
                    }
                }
            }
        }
    }

    /// Walk the whole float32 domain through every backend.
    ///
    /// Ignored by default: about a minute per kernel on four cores. Run with
    ///
    /// ```text
    /// cargo test -p engine --release --lib -- --ignored exhaustively
    /// ```
    fn sweep(op: Op, backend: Backend) -> (i64, u64) {
        use rayon::prelude::*;
        const BLOCK: u64 = 8192;
        const BLOCKS: u64 = (1u64 << 32) / BLOCK;
        (0..BLOCKS)
            .into_par_iter()
            .map(|b| {
                let base = b * BLOCK;
                let xs: Vec<f32> = (0..BLOCK)
                    .map(|k| f32::from_bits((base + k) as u32))
                    .collect();
                let mut out = vec![MaybeUninit::uninit(); xs.len()];
                apply(op, backend, &xs, &mut out);
                xs.iter()
                    .zip(out)
                    .fold((0i64, 0u64), |(worst, n), (&x, o)| {
                        // SAFETY: `apply` initialized every element.
                        let got = unsafe { o.assume_init() };
                        let want = reference(op, x);
                        if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
                            (worst, n)
                        } else {
                            (worst.max(ulps_apart(got, want)), n + 1)
                        }
                    })
            })
            .reduce(|| (0, 0), |a, b| (a.0.max(b.0), a.1 + b.1))
    }

    #[test]
    #[ignore = "sweeps all 2^32 float32 inputs; takes ~1 minute"]
    fn tanh_matches_promoted_reference_exhaustively() {
        for backend in available() {
            let (worst, differing) = sweep(Op::Tanh, backend);
            assert_eq!(
                differing, 0,
                "{backend:?}: {differing} of 2^32 differ, worst {worst} ulp"
            );
        }
    }

    /// `erf`'s claim is weaker than `tanh`'s and stated as a number: at most one
    /// ulp, on at most a couple of hundred of the 4.3 billion inputs. The bound
    /// is loose enough not to be a rounding-mode tripwire and tight enough that
    /// a real regression -- `libm::erff` misrounds 127.6 million -- fails it.
    #[test]
    #[ignore = "sweeps all 2^32 float32 inputs; takes ~1 minute"]
    fn erf_is_almost_always_correctly_rounded_exhaustively() {
        for backend in available() {
            let (worst, differing) = sweep(Op::Erf, backend);
            assert!(worst <= 1, "{backend:?}: worst error {worst} ulp");
            assert!(
                differing <= 500,
                "{backend:?}: {differing} of 2^32 not correctly rounded"
            );
        }
    }

    /// The two gradient kernels, which take an extra operand and so go through
    /// their own dispatch. Checked against float64 references written the
    /// cancellation-free way, for the same reason the forward ones are.
    #[derive(Clone, Copy, Debug)]
    enum BinOp {
        GeluErfBackward,
        GeluTanhBackward,
    }

    fn run2(op: BinOp, backend: Backend, xs: &[f32], gs: &[f32]) -> Vec<f32> {
        let k = F32Kernel(backend);
        let mut out = vec![MaybeUninit::uninit(); xs.len()];
        match op {
            BinOp::GeluErfBackward => k.gelu_erf_backward(xs, gs, &mut out),
            BinOp::GeluTanhBackward => k.gelu_tanh_backward(xs, gs, &mut out),
        }
        out.into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect()
    }

    fn reference2(op: BinOp, x: f32, g: f32) -> f32 {
        let xd = x as f64;
        let local = match op {
            BinOp::GeluErfBackward => {
                let cdf = 0.5 * libm::erfc(-xd * std::f64::consts::FRAC_1_SQRT_2);
                let pdf = (-0.5 * xd * xd).exp() * 0.3989422804014327;
                cdf + xd * pdf
            }
            BinOp::GeluTanhBackward => {
                let x2 = xd * xd;
                let v = 0.7978845608028654 * (xd + 0.044715 * xd * x2);
                // sigmoid(2v) is 0.5*(1 + tanh(v)); sech^2(v) is 4*s*(1-s).
                let s = 1.0 / (1.0 + (-2.0 * v).exp());
                let sech2 = 4.0 * s * (1.0 - s);
                s + 0.5 * xd * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2)
            }
        };
        (local * g as f64) as f32
    }

    #[test]
    fn gelu_backward_matches_the_analytic_derivative() {
        let xs: Vec<f32> = sample_inputs()
            .into_iter()
            .filter(|v| v.is_finite() && v.abs() < 1e30)
            .collect();
        // A non-uniform incoming gradient, so a kernel that ignored it or
        // mismatched the two operands' blocking would show up.
        let gs: Vec<f32> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| (i % 17) as f32 * 0.25 - 2.0 + x.signum() * 0.5)
            .collect();
        for op in [BinOp::GeluErfBackward, BinOp::GeluTanhBackward] {
            for backend in available() {
                for ((&x, &g), got) in xs.iter().zip(gs.iter()).zip(run2(op, backend, &xs, &gs)) {
                    let want = reference2(op, x, g);
                    assert!(
                        ulps_apart(got, want) <= 2,
                        "{op:?}/{backend:?}: f'({x:e})*{g:e} gave {got:e}, want {want:e}"
                    );
                }
            }
        }
    }

    /// The gradient's negative tail decays to zero rather than plateauing --
    /// the same failure the forward pass had, and it reaches the derivative
    /// through the same `1 + erf` / `1 + tanh` cancellation.
    #[test]
    fn gelu_backward_tail_decays_to_zero() {
        let xs: Vec<f32> = vec![-4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -20.0, -40.0];
        let gs = vec![1.0f32; xs.len()];
        for op in [BinOp::GeluErfBackward, BinOp::GeluTanhBackward] {
            for backend in available() {
                let got = run2(op, backend, &xs, &gs);
                for w in got.windows(2) {
                    assert!(
                        w[1].abs() < w[0].abs() || w[1] == 0.0,
                        "{op:?}/{backend:?}: stopped decaying: {got:?}"
                    );
                }
                assert_eq!(
                    *got.last().unwrap(),
                    0.0,
                    "{op:?}/{backend:?}: should underflow to zero"
                );
            }
        }
    }

    #[test]
    fn gelu_backward_propagates_nan() {
        for op in [BinOp::GeluErfBackward, BinOp::GeluTanhBackward] {
            for backend in available() {
                let got = run2(op, backend, &[f32::NAN, 1.0, 1.0], &[1.0, f32::NAN, 1.0]);
                assert!(got[0].is_nan(), "{op:?}/{backend:?}: NaN input");
                assert!(got[1].is_nan(), "{op:?}/{backend:?}: NaN gradient");
                assert!(got[2].is_finite());
            }
        }
    }
}
