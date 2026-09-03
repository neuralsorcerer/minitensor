// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Vectorized float32 kernels for `tanh`, `erf`, `erfc`, `expm1`, `sinh`,
//! `cosh`, `log`, `log1p`, `softplus`, `sigmoid`, `silu`, `sin`, `cos`, `tan`,
//! both GELU variants, and the GELU and SiLU gradients.
//!
//! These were the slowest things in the elementwise surface, all for the same
//! reason: a `libm` call per element, which no amount of rayon parallelism can
//! vectorize away. `tanh` measured 11.8x slower than a hand-vectorized SIMD
//! `tanh` at a million float32 elements and 31x at 4096, *after* spreading over four
//! cores; `erf` -- and so the exact GELU that every transformer uses -- ran at
//! 25 ns per element.
//!
//! The fix is to write each routine so that LLVM can vectorize it, then compile
//! it several times over -- once per instruction set -- and pick at runtime.
//! That requires the whole loop to live inside the multiversioned function, so
//! the entry points here take a block rather than an element (see
//! `ops::map::unary_map_blocks_threshold`, and its two-input sibling
//! `binary_map_blocks_threshold` for the gradients). They all share one
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
//! # expm1, sinh, cosh
//!
//! That recombination *is* `expm1`, so once `tanh` had it these were nearly
//! free -- [`expm1_reduced`] is shared by all four:
//!
//! ```text
//!     tanh(x) = u/(u + 2),              u = expm1(2x)
//!     expm1   = u
//!     sinh(x) = u(u + 2) / (2(u + 1)),  u = expm1(|x|)
//!     cosh(x) = (e + 1/e) / 2,          e = u + 1
//! ```
//!
//! `sinh` and `cosh` evaluate at `|x|` on purpose. The same `sinh` expression
//! at negative `x` divides by `u + 1`, which is `exp(x)` and has underflowed to
//! zero -- `sinh(-100)` would come back as -inf. On `|x|` the denominator never
//! drops below 1, and oddness restores the sign afterwards.
//!
//! # log, log1p, softplus
//!
//! The other half of the module, on its own reduction: `u = 2^k * m` with `m`
//! in `[0.6875, 1.375)`, then
//!
//! ```text
//!     log(u) = k*ln2 + 2*atanh(s),   s = (m - 1)/(m + 1),  s^2 <= 0.0343
//! ```
//!
//! `atanh` rather than a series in `m - 1` because only odd powers appear and
//! `s` is four times smaller, which is the difference between a degree-9
//! polynomial and about forty terms.
//!
//! The decomposition is branchless, and that matters more than it looks.
//! Folding `[sqrt2, 2)` down to keep `s` small is naturally a comparison, but
//! biasing the bits by `0x3fe6...` first makes the exponent field of the
//! difference *be* `k`, with no comparison at all. Written the obvious way the
//! branch mispredicted about half the time and cost more than the rest of the
//! kernel.
//!
//! [`log_core`] takes a correction term `c` and returns `log(u) + c/u`, which
//! is what makes `log1p` accurate rather than merely correct-looking. `1 + x`
//! rounds, and for small `x` it rounds away most of `x`: at `x = 1e-10` a naive
//! `log(1 + x)` keeps six digits. Passing `c = x - ((1 + x) - 1)` -- exact, by
//! Sterbenz -- restores them. `log` itself passes no correction, and the
//! division compiles out entirely.
//!
//! `softplus` is `log1p(exp(beta*x))/beta`, and goes through `exp` rather than
//! `expm1` for the same class of reason: `log(2 + expm1(v))` is algebraically
//! equal and numerically useless, because for very negative `v` it forms
//! `2 + (-1 + tiny)` and rounds the tiny part away, leaving the tail 0.2%
//! wrong.
//!
//! # sigmoid, silu
//!
//! `sigmoid(x) = e/(e + 1)` with `e = exp(x)`, and `silu(x) = x * sigmoid(x)`.
//!
//! The scalar forms these replace were `1/(1 + exp(-x))`, which overflows:
//! below about `x = -89`, `exp(-x)` is infinite in float32 and the result
//! collapses to exactly zero while the true value is still representable.
//! `silu(-100)` returned -0 against -3.72e-42, and the softplus gradient --
//! which *is* sigmoid -- returned 0 at `x = -95` against 5.52e-42. `e/(e + 1)`
//! never forms that reciprocal.
//!
//! [`logistic_parts`] returns `1 - sigmoid(x)` alongside, as `1/(e + 1)`,
//! because the derivatives want it and forming it by subtraction throws the
//! tail away: at `x = 40` the true value is 4e-18 and `1 - s` in float64 is
//! exactly 0. The reciprocal is shared, so it is free.
//!
//! # sin, cos, tan
//!
//! A third reduction, and the only one here that cannot cover its whole input
//! range: `x = n*(pi/2) + r` with `|r| <= pi/4`, then `sin(r)/r` and `cos(r)`
//! as polynomials in `r^2` and a quadrant select on `n & 3`. `tan` is the same
//! reduction plus a division, which is why it comes nearly free and is the
//! biggest speedup of the three.
//!
//! `pi/2` is split three ways, each part carrying at most 33 significant bits,
//! so `n * PIO2_k` is exact for every `|n| < 2^20`. That exactness is the whole
//! game: at `x ~ 2^20` the subtraction `x - n*(pi/2)` discards 20 bits, and
//! there has to be something below them or the result is noise.
//!
//! Past `2^20` there is nothing below them, and the honest fix is to stop:
//! [`trig_block_kernel`] runs a second pass that redoes those elements with the
//! scalar float64 routine, which reduces properly. It is a separate pass rather
//! than a branch in the element function because a `libm` call inside the main
//! loop would scalarize all of it, for inputs that essentially never occur.
//!
//! One subtlety worth recording: `sin` is *not* sign-preserving, so the
//! `copysign` repair the other kernels use for `-0.0` would be wrong here
//! (`sin(4)` is negative). The cause was `fma(r*t, ps, r)` -- at `r = -0.0`,
//! `ps` is negative, so `-0.0 * ps` is `+0.0` and the sum is `+0.0`. Writing it
//! as `r * (1 + t*ps)` keeps the sign for free, and was the difference between
//! one differing input and none.
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
//! relative error 5.9e-08 -- carries over unchanged.
//! Its polynomial degree was picked against that same sweep: one term shorter
//! still matches everywhere, two terms shorter breaks 43 inputs, so degree 12
//! is the first with a whole term of margin.
//!
//! `expm1`, `sinh` and `cosh` are bit-identical on all 2^32 inputs too.
//! `expm1` and `sinh` replaced promoted scalars, so that preserves what they
//! already returned; `cosh` replaced glibc's `coshf`, which misrounds
//! 22,628,918 of the 2^32 inputs (0.527%), so there it is an accuracy gain as
//! well as a speedup.
//!
//! `sin`, `cos` and `tan` are bit-identical on all 2^32 inputs as well, the
//! fallback pass included -- the handover at 2^20 is not visible in the output.
//!
//! `log` is bit-identical too, where the `f32::ln` it replaces misrounds
//! 416,909 of the 2^32 inputs (0.0097%). `log1p` misses exactly one input by
//! one ulp.
//!
//! `erf` and `erfc` are within one ulp of the correctly rounded result
//! everywhere, and are *the* correctly rounded result on all but 68 and 131,334
//! of the 2^32 inputs respectively. The routines they replace misround
//! 127,576,760 (`erff`, 2.97%) and 19,954,784 (`erfcf`, 0.465%), so both are
//! large accuracy gains as well as speedups. `erfc` misses more often than
//! `erf` because below `|x| = 2` it does have to form `1 - erf`, which costs
//! 7.7 bits at `x = 2`; above 2, where it matters, it reads the `erfc` branch
//! directly and forms no difference at all.
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
//!     erfc               5347us    651us    8.2x
//!     expm1              4696us    283us   16.6x
//!     sinh               6091us    410us   14.8x
//!     cosh               2138us    358us    6.0x
//!     gelu (exact)       6471us    953us    6.8x
//!     gelu (tanh)        9340us    655us   14.3x
//!     gelu backward      7937us   2372us    3.3x
//!     log                 895us    438us    2.0x
//!     log1p              4257us    453us    9.4x
//!     softplus           7388us   1031us    7.2x
//!     sigmoid            2541us    640us    4.0x
//!     silu               1888us    687us    2.7x
//!     silu backward      ~2.1ms   ~0.7ms    ~3x
//!     sin                2538us    515us    4.9x
//!     cos                2632us    457us    5.8x
//!     tan                6319us    544us   11.6x
//! ```
//!
//! The backward figure is the whole gradient step, so about 1.3ms of it is
//! autograd graph and allocation overhead that this work does not touch -- the
//! kernel itself went from roughly 6.6ms to 1.1ms. It was the most expensive
//! gradient in the activation set, costing more than the forward pass.
//!
//! `tanh` went from 11.8x slower than a hand-vectorized SIMD `tanh` to ahead
//! of it. At small sizes a gap remains, and it is the accuracy decision
//! showing up as a cost: evaluating in float32 lanes (16 wide under AVX-512)
//! with a shorter polynomial beats these float64 lanes (8 wide).
//!
//! # The portable fallback
//!
//! One honest caveat. On a host with neither AVX-512 nor AVX2+FMA the loops
//! still compile, but to SSE2 without fused multiply-add, and the log kernel
//! measures about 10 ns/element against `f32::ln`'s 4 -- it is latency-bound on
//! a division and a ten-deep polynomial that the vector paths hide across
//! lanes. Every other kernel here is still ahead of what it replaced on that
//! path.
//!
//! It is kept anyway, rather than falling back to the scalar routine, because
//! every kernel in this module returns identical bits on every backend, and a
//! result that depends on which CPU it ran on is worth more than the throughput
//! of a fallback no x86-64 part since 2013 selects.
//!
//! # Not covered
//!
//! The float64 kernels all still call `libm`. The same skeleton would serve
//! them, but a correctly-rounded float64 result needs the reduction and the
//! polynomials carried to ~2^-60 -- double-double residuals in places where
//! the float32 path can round freely -- and that is a different piece of work.
//! The float64 gap is also much smaller to begin with (`tanh` is 2.5x off a
//! vectorized baseline at a million elements against float32's 11.8x).
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

/// `exp(v) - 1` for an already-bounded `v`, recombined from the reduction.
///
/// This is the whole of `expm1`, and every other kernel in this module is built
/// on it: `tanh` is `u/(u+2)`, `sinh` is `u(u+2)/(2(u+1))`, `cosh` follows from
/// the same `u`. Splitting it out is why they cost so little to add.
#[inline(always)]
fn expm1_reduced<const FMA: bool>(v: f64) -> f64 {
    let (two_pow_n, r) = reduce_exp::<FMA>(v);
    fma_or::<FMA>(two_pow_n, expm1_poly::<FMA>(r), two_pow_n - 1.0)
}

/// `tanh` in float64. Branch-free by construction so callers vectorize.
#[inline(always)]
fn tanh_core<const FMA: bool>(xd: f64) -> f64 {
    // `clamp` leaves NaN alone (both of its comparisons fail), which is what
    // carries a NaN input through to a NaN result.
    let t = xd.clamp(-TANH_LIMIT, TANH_LIMIT) * 2.0;
    let u = expm1_reduced::<FMA>(t);
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

/// `erfc` in float64.
///
/// `erfc(x)` is `1 + erf(-x)`, so this is [`one_plus_erf`] at `-x` and inherits
/// its accuracy: above 2 the value comes from the `erfc` branch of
/// [`erf_parts`] with no subtraction at all, which is exactly the regime
/// `erfc` exists to serve. The clamp reaches far enough -- `erfc` underflows
/// float32 at about 10.2, inside the fitted range.
#[inline(always)]
fn erfc_core<const FMA: bool>(xd: f64) -> f64 {
    one_plus_erf::<FMA>(-xd)
}

#[inline(always)]
fn erfc_one<const FMA: bool>(x: f32) -> f32 {
    erfc_core::<FMA>(x as f64) as f32
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

// ---------------------------------------------------------------------------
// exp family: expm1, sinh, cosh
// ---------------------------------------------------------------------------

/// Clamp for the exp-family kernels.
///
/// `exp(100)` is 2.7e43, already well past float32's 3.4e38, so everything the
/// clamp discards converts to infinity anyway; and `expm1(-100)` is -1 exactly
/// in float64, which is also what the true value rounds to. It exists only to
/// keep `2^n` inside an exponent field for absurd inputs.
const EXP_FAMILY_LIMIT: f64 = 100.0;

/// Clamp for `exp` itself, which needs a wider one than its relatives.
///
/// `exp(-100)` is 3.7e-44 -- a float32 subnormal, and a *different* one from
/// `exp(-101)`, so clamping at 100 would round a range of distinct answers
/// together. Float32 runs out below `exp(-104)`, so 110 is the first round
/// number where everything discarded converts to zero anyway. On the positive
/// side anything past 88.7 is already infinity.
const EXP_LIMIT: f64 = 110.0;

/// `exp` in float64.
///
/// The same reduction `expm1` uses, without its final `- 1`: with
/// `x = n*ln2 + r`, `exp(x)` is `2^n * (1 + expm1_poly(r))`, and the `2^n`
/// factors out of the fused multiply-add.
#[inline(always)]
fn exp_core<const FMA: bool>(xd: f64) -> f64 {
    let (two_pow_n, r) = reduce_exp::<FMA>(xd.clamp(-EXP_LIMIT, EXP_LIMIT));
    fma_or::<FMA>(two_pow_n, expm1_poly::<FMA>(r), two_pow_n)
}

/// `exp` for one float32.
#[inline(always)]
fn exp_one<const FMA: bool>(x: f32) -> f32 {
    exp_core::<FMA>(x as f64) as f32
}

/// `exp(x - shift)` for one float32, the pass `softmax` spends about half its
/// time in.
///
/// The shift is what keeps `exp` from overflowing on a large input, so it can
/// never be folded away; taking it as a parameter is what lets the vectorized
/// kernel do the subtraction too, rather than handing back a buffer for a
/// scalar loop to subtract from. It also subtracts in float64, where
/// `softmax`'s own loop subtracted in float32 and rounded twice.
#[inline(always)]
fn exp_shifted_one<const FMA: bool>(x: f32, shift: f64, _unused: f64) -> f32 {
    exp_core::<FMA>(x as f64 - shift) as f32
}

/// `expm1` in float64.
#[inline(always)]
fn expm1_core<const FMA: bool>(xd: f64) -> f64 {
    let y = expm1_reduced::<FMA>(xd.clamp(-EXP_FAMILY_LIMIT, EXP_FAMILY_LIMIT));
    // `expm1` carries the sign of its argument, so the same signed-zero repair
    // as `tanh` applies: the polynomial turns `-0.0` into `+0.0`.
    y.copysign(xd)
}

/// `sinh` in float64, as `u(u+2) / (2(u+1))` with `u = expm1(|x|)`.
///
/// Evaluated at `|x|` rather than `x` on purpose. For large negative `x` the
/// same expression divides by `u + 1`, which is `exp(x)` and has underflowed to
/// zero -- `sinh(-100)` would come back as -inf. On `|x|` the denominator is
/// never below 1, and oddness restores the sign at the end.
#[inline(always)]
fn sinh_core<const FMA: bool>(xd: f64) -> f64 {
    let a = xd.abs().clamp(0.0, EXP_FAMILY_LIMIT);
    let u = expm1_reduced::<FMA>(a);
    let y = u * (u + 2.0) / (2.0 * (u + 1.0));
    y.copysign(xd)
}

/// `cosh` in float64, as `(e + 1/e) / 2` with `e = exp(|x|)` reconstructed from
/// the same `u`. Even, so no sign repair.
#[inline(always)]
fn cosh_core<const FMA: bool>(xd: f64) -> f64 {
    let a = xd.abs().clamp(0.0, EXP_FAMILY_LIMIT);
    let e = expm1_reduced::<FMA>(a) + 1.0;
    0.5 * (e + 1.0 / e)
}

#[inline(always)]
fn expm1_one<const FMA: bool>(x: f32) -> f32 {
    expm1_core::<FMA>(x as f64) as f32
}

#[inline(always)]
fn sinh_one<const FMA: bool>(x: f32) -> f32 {
    sinh_core::<FMA>(x as f64) as f32
}

#[inline(always)]
fn cosh_one<const FMA: bool>(x: f32) -> f32 {
    cosh_core::<FMA>(x as f64) as f32
}

// ---------------------------------------------------------------------------
// log family: log, log1p, softplus
// ---------------------------------------------------------------------------

const R0: f64 = 1.0 / 3.0;
const R1: f64 = 1.0 / 5.0;
const R2: f64 = 1.0 / 7.0;
const R3: f64 = 1.0 / 9.0;
const R4: f64 = 1.0 / 11.0;
const R5: f64 = 1.0 / 13.0;
const R6: f64 = 1.0 / 15.0;
const R7: f64 = 1.0 / 17.0;
const R8: f64 = 1.0 / 19.0;
const R9: f64 = 1.0 / 21.0;

/// `log(u) + c/u`. `c` carries the residual of a `1 + x` that rounded.
#[inline(always)]
fn log_core<const FMA: bool, const CORRECT: bool>(u: f64, c: f64) -> f64 {
    // Branchless decompose u = 2^k * m with m in [0.6875, 1.375). Biasing by
    // OFF first makes the exponent field of the difference *be* k, so no
    // data-dependent branch is needed to fold the [sqrt2, 2) half down -- that
    // branch mispredicted about half the time and cost more than the rest of
    // the kernel on the scalar path.
    const OFF: u64 = 0x3fe6_0000_0000_0000;
    let bits = u.to_bits();
    let tmp = bits.wrapping_sub(OFF);
    let k = ((tmp as i64) >> 52) as i32;
    let m = f64::from_bits(bits.wrapping_sub(tmp & (0xfff << 52)));
    let s = (m - 1.0) / (m + 1.0);
    let t = s * s;
    // Estrin over R(t) = 1/3 + t/5 + t^2/7 + ...
    let t2 = t * t;
    let t4 = t2 * t2;
    let a0 = fma_or::<FMA>(t, R1, R0);
    let a1 = fma_or::<FMA>(t, R3, R2);
    let a2 = fma_or::<FMA>(t, R5, R4);
    let a3 = fma_or::<FMA>(t, R7, R6);
    let a4 = fma_or::<FMA>(t, R9, R8);
    let b0 = fma_or::<FMA>(t2, a1, a0);
    let b1 = fma_or::<FMA>(t2, a3, a2);
    let r = fma_or::<FMA>(t4, fma_or::<FMA>(t4, a4, b1), b0);
    // log(m) = 2s(1 + t R(t)); the leading `s` stays outside so relative
    // accuracy survives m -> 1.
    let log_m = 2.0 * fma_or::<FMA>(s * t, r, s);
    let kf = k as f64;
    // `log` passes no correction, so its division is compiled away entirely --
    // it is the single most expensive instruction in the kernel.
    let corr = if CORRECT { c / u } else { 0.0 };
    let y = fma_or::<FMA>(kf, LN2_HI, log_m + fma_or::<FMA>(kf, LN2_LO, corr));

    // u == 0 -> -inf; u < 0 or NaN -> NaN; u == +inf -> +inf.
    if u > 0.0 {
        if u.is_finite() { y } else { u }
    } else if u == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn log_one<const FMA: bool>(x: f32) -> f32 {
    log_core::<FMA, false>(x as f64, 0.0) as f32
}

#[inline(always)]
fn log1p_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let u = 1.0 + xd;
    // The exact residual of the sum: `u - 1` is exact by Sterbenz for the
    // |x| <= 1 that matters, so `c` is precisely what `1 + x` rounded away.
    // Without it `log1p(1e-10)` keeps only six digits -- the whole point of a
    // separate `log1p` is that `log(1 + x)` cannot be formed naively.
    let c = xd - (u - 1.0);
    // `log1p` is increasing through zero, so it carries the sign of its
    // argument; the arithmetic loses that at `x = -0.0`, where `c` is `-0.0`
    // and `-0.0/1.0` gets added into a `+0.0`. Same repair as `tanh` and
    // `expm1`, and a no-op for every other input.
    (log_core::<FMA, true>(u, c) as f32).copysign(x)
}

/// Lower clamp on `beta * x` inside softplus. `exp(-300)` is 5e-131 and the
/// result is that same value divided by `beta`, so everything below has long
/// since underflowed float32.
const SOFTPLUS_LIMIT: f64 = 300.0;

/// `log1p(exp(beta*x)) / beta`, with the large-`x` linear tail taken by the
/// caller's threshold.
///
/// Goes through `exp` rather than `expm1` on purpose. `log(2 + expm1(v))` looks
/// equivalent and is not: for very negative `v` it forms `2 + (-1 + tiny)`,
/// which rounds the tiny part away and leaves the tail 0.2% wrong. `exp(v)` as
/// `2^n(1 + p)` never cancels, and `log1p` then keeps it.
#[inline(always)]
fn softplus_one<const FMA: bool>(x: f32, beta: f64, threshold: f64) -> f32 {
    let xd = x as f64;
    let v = beta * xd;
    let (two_pow_n, r) = reduce_exp::<FMA>(v.max(-SOFTPLUS_LIMIT));
    let w = two_pow_n * (1.0 + expm1_poly::<FMA>(r));
    let u = 1.0 + w;
    let c = w - (u - 1.0);
    let soft = log_core::<FMA, true>(u, c) / beta;
    // Above the threshold softplus is `x` to within float32; NaN fails the
    // comparison and comes through the computed side, which is NaN too.
    if v > threshold {
        xd as f32
    } else {
        soft as f32
    }
}

/// `-log(sigmoid(x))`, which is `softplus(-x)`.
///
/// The quantity binary cross-entropy is written in terms of, and `logsigmoid`
/// negated. Both used to reach it through two scalar `libm` calls per element
/// -- an `exp` and a `ln_1p` -- while `softplus` itself had this kernel all
/// along.
///
/// The threshold is `softplus`'s own default. Above it the answer is `-x` to
/// well within float32: the term dropped at the boundary is `log1p(exp(-20))`,
/// or 2.1e-9, against an ulp of 1.9e-6 at that magnitude. It is not an
/// optimisation but a necessity -- `exp(-x)` for a large negative `x`
/// overflows even float64, and the linear branch is what the shift protects.
#[inline(always)]
fn neg_log_sigmoid_one<const FMA: bool>(x: f32) -> f32 {
    softplus_one::<FMA>(-x, 1.0, SOFTPLUS_THRESHOLD)
}

/// The threshold above which `softplus(z)` is `z` to within float32.
pub(crate) const SOFTPLUS_THRESHOLD: f64 = 20.0;

// ---------------------------------------------------------------------------
// logistic family: sigmoid, silu
// ---------------------------------------------------------------------------

/// Clamp for the logistic kernels. `sigmoid` saturates to 1.0 in float32 by
/// `x = 17` and underflows by `x = -104`, and `x*sigmoid(x)` underflows by
/// about -110, so 300 is far outside anything observable; it only keeps `2^n`
/// inside an exponent field.
const LOGISTIC_LIMIT: f64 = 300.0;

/// `(sigmoid(x), 1 - sigmoid(x))`, both from one `exp` and one reciprocal.
///
/// Returned as a pair because `1 - sigmoid(x)` is what the derivatives want and
/// forming it by subtraction throws away the tail: at `x = 40` the true value
/// is 4e-18 and `1 - s` in float64 is exactly 0. As `1/(e + 1)` it is simply
/// correct, and costs nothing extra -- the reciprocal is shared.
#[inline(always)]
fn logistic_parts<const FMA: bool>(xd: f64) -> (f64, f64) {
    let (two_pow_n, r) = reduce_exp::<FMA>(xd.clamp(-LOGISTIC_LIMIT, LOGISTIC_LIMIT));
    let e = two_pow_n * (1.0 + expm1_poly::<FMA>(r));
    let recip = 1.0 / (e + 1.0);
    (e * recip, recip)
}

#[inline(always)]
fn sigmoid_one<const FMA: bool>(x: f32) -> f32 {
    logistic_parts::<FMA>(x as f64).0 as f32
}

/// `x * sigmoid(x)`.
///
/// The scalar form this replaces was `x / (1 + exp(-x))`, which overflows: at
/// `x = -100`, `exp(-x)` is infinite in float32, so it returned `-0` where
/// -3.72e-42 is perfectly representable. `e/(e+1)` never forms that reciprocal
/// and gets the tail right.
#[inline(always)]
fn silu_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let (s, _) = logistic_parts::<FMA>(xd);
    // Below the clamp the product has underflowed float32 anyway; zeroing keeps
    // `x = -inf` at NaN, which is what the scalar form returned.
    let s = if xd < -LOGISTIC_LIMIT { 0.0 } else { s };
    (xd * s) as f32
}

/// `d/dx [x*sigmoid(x)] = s * (1 + x*(1 - s))`, times the incoming gradient.
#[inline(always)]
fn silu_backward_one<const FMA: bool>(x: f32, gout: f32) -> f32 {
    let xd = x as f64;
    let (s, one_minus_s) = logistic_parts::<FMA>(xd);
    let (s, one_minus_s) = if xd < -LOGISTIC_LIMIT {
        (0.0, 1.0)
    } else {
        (s, one_minus_s)
    };
    let local = s * fma_or::<FMA>(xd, one_minus_s, 1.0);
    (local * gout as f64) as f32
}

// ---------------------------------------------------------------------------
// trig: sin, cos, tan
// ---------------------------------------------------------------------------

/// `2/pi`, and `pi/2` split three ways.
///
/// Each part carries at most 33 significant bits, so `n * PIO2_k` is exact for
/// every `|n| < 2^20` the reduction produces, and the three together represent
/// `pi/2` to within 1e-37. That is what lets `x - n*pi/2` be computed without
/// losing the cancellation: at `x ~ 2^20` the subtraction discards 20 bits, and
/// there has to be something below them.
const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
const PIO2_0: f64 = 1.5707963267341256;
const PIO2_1: f64 = 6.077100506303966e-11;
const PIO2_2: f64 = 2.0222662487959506e-21;

/// Above this the three-part split runs out of exactness. Those elements are
/// redone with the scalar float64 routine, which reduces properly
/// (Payne-Hanek) -- see [`trig_block_kernel`].
const TRIG_LIMIT: f32 = 1048576.0; // 2^20

/// `sin(r)/r` and `cos(r)` for |r| <= pi/4, as polynomials in `t = r*r`.
#[inline(always)]
fn sin_cos_poly<const FMA: bool>(r: f64, t: f64) -> (f64, f64) {
    const S1: f64 = -1.0 / 6.0;
    const S2: f64 = 1.0 / 120.0;
    const S3: f64 = -1.0 / 5040.0;
    const S4: f64 = 1.0 / 362880.0;
    const S5: f64 = -1.0 / 39916800.0;
    const S6: f64 = 1.0 / 6227020800.0;
    const S7: f64 = -1.0 / 1307674368000.0;
    const S8: f64 = 1.0 / 355687428096000.0;
    const C1: f64 = -1.0 / 2.0;
    const C2: f64 = 1.0 / 24.0;
    const C3: f64 = -1.0 / 720.0;
    const C4: f64 = 1.0 / 40320.0;
    const C5: f64 = -1.0 / 3628800.0;
    const C6: f64 = 1.0 / 479001600.0;
    const C7: f64 = -1.0 / 87178291200.0;
    const C8: f64 = 1.0 / 20922789888000.0;
    const C9: f64 = -1.0 / 6402373705728000.0;

    let t2 = t * t;
    let t4 = t2 * t2;
    // sin: r * (1 + t*Ps(t)), Ps degree 7, Estrin
    let a0 = fma_or::<FMA>(t, S2, S1);
    let a1 = fma_or::<FMA>(t, S4, S3);
    let a2 = fma_or::<FMA>(t, S6, S5);
    let a3 = fma_or::<FMA>(t, S8, S7);
    let ps = fma_or::<FMA>(t4, fma_or::<FMA>(t2, a3, a2), fma_or::<FMA>(t2, a1, a0));
    let s = r * fma_or::<FMA>(t, ps, 1.0);
    // cos: 1 + t*Pc(t), Pc degree 8, Estrin
    let b0 = fma_or::<FMA>(t, C2, C1);
    let b1 = fma_or::<FMA>(t, C4, C3);
    let b2 = fma_or::<FMA>(t, C6, C5);
    let b3 = fma_or::<FMA>(t, C8, C7);
    let pc = fma_or::<FMA>(
        t4,
        fma_or::<FMA>(t4, C9, fma_or::<FMA>(t2, b3, b2)),
        fma_or::<FMA>(t2, b1, b0),
    );
    let c = fma_or::<FMA>(t, pc, 1.0);
    (s, c)
}

/// `(sin(r)/., cos(r), quadrant)` after reducing `x` mod pi/2.
#[inline(always)]
fn reduce_trig<const FMA: bool>(xd: f64) -> (f64, f64, u64) {
    let z = fma_or::<FMA>(xd, TWO_OVER_PI, MAGIC);
    let n = z - MAGIC;
    let q = z.to_bits() & 3;
    let r = fma_or::<FMA>(
        -n,
        PIO2_2,
        fma_or::<FMA>(-n, PIO2_1, fma_or::<FMA>(-n, PIO2_0, xd)),
    );
    let (s, c) = sin_cos_poly::<FMA>(r, r * r);
    (s, c, q)
}

#[inline(always)]
fn sin_one<const FMA: bool>(x: f32) -> f32 {
    let (s, c, q) = reduce_trig::<FMA>(x as f64);
    let v = if q & 1 != 0 { c } else { s };
    (if q & 2 != 0 { -v } else { v }) as f32
}

#[inline(always)]
fn cos_one<const FMA: bool>(x: f32) -> f32 {
    let (s, c, q) = reduce_trig::<FMA>(x as f64);
    let v = if q & 1 != 0 { s } else { c };
    // cos leads sin by one quadrant: negate on q in {1,2}
    (if ((q + 1) & 2) != 0 { -v } else { v }) as f32
}

#[inline(always)]
fn tan_one<const FMA: bool>(x: f32) -> f32 {
    let (s, c, q) = reduce_trig::<FMA>(x as f64);
    (if q & 1 != 0 { -c / s } else { s / c }) as f32
}

// ---------------------------------------------------------------------------
// arctangent
// ---------------------------------------------------------------------------

/// The breakpoints the reduction folds around, `tan(pi/8)` and `tan(3pi/8)`.
///
/// They are what makes one polynomial cover the whole line: every argument
/// lands inside `[-tan(pi/8), tan(pi/8)]` after the fold, where a short
/// series is enough.
const ATAN_LO: f64 = 0.414_213_562_373_095_1;
const ATAN_HI: f64 = 2.414_213_562_373_095;

/// `atan(w)/w` for `u = w^2` in `[0, tan(pi/8)^2]`.
///
/// Chebyshev-fitted rather than truncated from the Taylor series: the series
/// needs twenty-four terms for this accuracy on this interval and the fit
/// needs eleven. Worst error over the interval is 1.3e-18, three orders below
/// a float64 ulp, so the float32 handed back is the correctly rounded one.
///
/// The leading `w` stays outside, as it does in `expm1_poly`: folding it in
/// would round it against terms it dominates as `w -> 0`.
#[inline(always)]
fn atan_poly<const FMA: bool>(u: f64) -> f64 {
    let mut p = -0.017_805_397_205_419_446;
    p = fma_or::<FMA>(p, u, 0.037_965_257_453_865_93);
    p = fma_or::<FMA>(p, u, -0.050_351_024_566_015_52);
    p = fma_or::<FMA>(p, u, 0.058_468_782_973_308_72);
    p = fma_or::<FMA>(p, u, -0.066_629_518_136_291_91);
    p = fma_or::<FMA>(p, u, 0.076_920_453_309_022_25);
    p = fma_or::<FMA>(p, u, -0.090_908_968_090_640_27);
    p = fma_or::<FMA>(p, u, 0.111_111_107_449_196_58);
    p = fma_or::<FMA>(p, u, -0.142_857_142_792_502_45);
    p = fma_or::<FMA>(p, u, 0.199_999_999_999_408_93);
    p = fma_or::<FMA>(p, u, -0.333_333_333_333_331_2);
    fma_or::<FMA>(p, u, 1.0)
}

/// `atan` for one float32.
///
/// One division for the whole reduction. The three cases each want a
/// different quotient -- `-1/a` above `tan(3pi/8)`, `(a-1)/(a+1)` above
/// `tan(pi/8)`, and `a` itself below -- so the numerator and denominator are
/// selected and divided once, rather than dividing inside each branch and
/// making the vectorized form pay for all of them.
///
/// No special cases. An infinity divides to a signed zero and leaves the
/// `pi/2` the fold added; a NaN fails both comparisons and comes through the
/// arithmetic as a NaN; and `copysign` rather than a test on the sign is what
/// carries `-0.0` through to `-0.0`.
#[inline(always)]
fn atan_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let a = xd.abs();
    let big = a > ATAN_HI;
    let mid = a > ATAN_LO;

    let numerator = if big {
        -1.0
    } else if mid {
        a - 1.0
    } else {
        a
    };
    let denominator = if big {
        a
    } else if mid {
        a + 1.0
    } else {
        1.0
    };
    let base = if big {
        std::f64::consts::FRAC_PI_2
    } else if mid {
        std::f64::consts::FRAC_PI_4
    } else {
        0.0
    };

    let w = numerator / denominator;
    let folded = fma_or::<FMA>(w, atan_poly::<FMA>(w * w), base);
    (folded.copysign(xd)) as f32
}

// ---------------------------------------------------------------------------
// arcsine and arccosine
// ---------------------------------------------------------------------------

/// `(asin(t) - t) / t^3` for `u = t^2` in `[0, 1/4]`.
///
/// One polynomial serves the whole domain because both branches of the
/// reduction land in that interval: see [`asin_half`]. Chebyshev-fitted to a
/// worst error of 9.8e-19 over it.
///
/// The leading `t` stays outside, as it does in `atan_poly`: folding it in
/// would round it against terms it dominates as `t -> 0`.
#[inline(always)]
fn asin_poly<const FMA: bool>(u: f64) -> f64 {
    let mut p = 0.029_612_011_264_955_12;
    p = fma_or::<FMA>(p, u, -0.019_241_671_746_743_04);
    p = fma_or::<FMA>(p, u, 0.019_554_513_336_123_378);
    p = fma_or::<FMA>(p, u, 0.003_044_879_909_455_677_3);
    p = fma_or::<FMA>(p, u, 0.009_319_560_794_767_446);
    p = fma_or::<FMA>(p, u, 0.009_621_842_970_100_282);
    p = fma_or::<FMA>(p, u, 0.011_566_459_612_121_669);
    p = fma_or::<FMA>(p, u, 0.013_963_780_012_203_57);
    p = fma_or::<FMA>(p, u, 0.017_352_816_540_325_496);
    p = fma_or::<FMA>(p, u, 0.022_372_157_443_507_22);
    p = fma_or::<FMA>(p, u, 0.030_381_944_475_532_34);
    p = fma_or::<FMA>(p, u, 0.044_642_857_142_551_895);
    p = fma_or::<FMA>(p, u, 0.075_000_000_000_001_18);
    fma_or::<FMA>(p, u, 0.166_666_666_666_666_66)
}

/// The reduction `asin` and `acos` share: the series value `v` and which
/// branch produced it.
///
/// Below a half the series is evaluated at `|x|` itself. Above it the
/// half-angle identity `asin(a) = pi/2 - 2*asin(sqrt((1-a)/2))` moves the
/// argument back down -- and `(1-a)/2` is at most a quarter there, which is
/// exactly the interval `x^2` occupies on the other branch. That is why one
/// polynomial covers both, and why the square root is the only extra
/// operation the fold costs.
///
/// An `|x|` above one makes `(1-a)/2` negative, so the square root is NaN and
/// the NaN carries through -- which is the answer outside the domain, with no
/// test for it.
#[inline(always)]
fn asin_half<const FMA: bool>(a: f64) -> (f64, bool) {
    let big = a > 0.5;
    let u = if big { (1.0 - a) * 0.5 } else { a * a };
    let s = if big { u.sqrt() } else { a };
    // `s * u` is `s^3` either way: above the fold `u` is `s^2` by
    // construction, and below it `u` is `a^2` and `s` is `a`.
    (fma_or::<FMA>(s * u, asin_poly::<FMA>(u), s), big)
}

#[inline(always)]
fn asin_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let (v, big) = asin_half::<FMA>(xd.abs());
    let r = if big {
        std::f64::consts::FRAC_PI_2 - 2.0 * v
    } else {
        v
    };
    (r.copysign(xd)) as f32
}

/// `acos`, taken from the same reduction rather than as `pi/2 - asin(x)`.
///
/// That subtraction is where `acos` loses its accuracy near `x = 1`: both
/// sides are close to `pi/2` and the difference is small, so the leading
/// digits cancel. The half-angle branch hands back the small answer directly
/// as `2v`, with nothing to cancel.
#[inline(always)]
fn acos_one<const FMA: bool>(x: f32) -> f32 {
    let xd = x as f64;
    let (v, big) = asin_half::<FMA>(xd.abs());
    let r = if big {
        if xd < 0.0 {
            std::f64::consts::PI - 2.0 * v
        } else {
            2.0 * v
        }
    } else {
        std::f64::consts::FRAC_PI_2 - v.copysign(xd)
    };
    r as f32
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

/// The trig form: the vectorized loop, then a second pass redoing the rare
/// elements the reduction cannot handle.
///
/// The fallback has to be a separate pass rather than a branch inside the
/// element function. A `libm` call in the main loop would scalarize all of it,
/// for inputs that essentially never occur -- the second pass is a compare per
/// element over a block still in cache, and its body is almost never taken.
macro_rules! trig_block_kernel {
    ($block:ident, $one:ident, $scalar:path, $avx512:ident, $avx2:ident) => {
        #[inline(always)]
        fn $block<const FMA: bool>(input: &[f32], out: &mut [MaybeUninit<f32>]) {
            debug_assert_eq!(input.len(), out.len());
            for (o, &x) in out.iter_mut().zip(input.iter()) {
                o.write($one::<FMA>(x));
            }
            for (o, &x) in out.iter_mut().zip(input.iter()) {
                if x.abs() >= TRIG_LIMIT {
                    o.write($scalar(x as f64) as f32);
                }
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

/// The parameterized form, for kernels that take runtime scalars. The
/// parameters are captured once per block, never per element.
macro_rules! block_kernel_param {
    ($block:ident, $one:ident, $avx512:ident, $avx2:ident) => {
        #[inline(always)]
        fn $block<const FMA: bool>(input: &[f32], out: &mut [MaybeUninit<f32>], a: f64, b: f64) {
            debug_assert_eq!(input.len(), out.len());
            for (o, &x) in out.iter_mut().zip(input.iter()) {
                o.write($one::<FMA>(x, a, b));
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        fn $avx512(input: &[f32], out: &mut [MaybeUninit<f32>], a: f64, b: f64) {
            $block::<true>(input, out, a, b)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        fn $avx2(input: &[f32], out: &mut [MaybeUninit<f32>], a: f64, b: f64) {
            $block::<true>(input, out, a, b)
        }
    };
}

/// Dispatch for [`block_kernel_param!`].
macro_rules! dispatch_param {
    ($self:expr, $input:expr, $out:expr, $a:expr, $b:expr,
     $block:ident, $avx512:ident, $avx2:ident) => {
        match $self.0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `select` returned this variant only after
            // `is_x86_feature_detected!` confirmed avx512f on this CPU.
            Backend::Avx512 => unsafe { $avx512($input, $out, $a, $b) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above, for avx2 and fma.
            Backend::Avx2Fma => unsafe { $avx2($input, $out, $a, $b) },
            #[cfg(target_arch = "aarch64")]
            Backend::NativeFma => $block::<true>($input, $out, $a, $b),
            Backend::Portable => $block::<false>($input, $out, $a, $b),
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
block_kernel!(erfc_block, erfc_one, erfc_block_avx512, erfc_block_avx2);
block_kernel!(log_block, log_one, log_block_avx512, log_block_avx2);
block_kernel!(log1p_block, log1p_one, log1p_block_avx512, log1p_block_avx2);
block_kernel!(
    sigmoid_block,
    sigmoid_one,
    sigmoid_block_avx512,
    sigmoid_block_avx2
);
block_kernel!(silu_block, silu_one, silu_block_avx512, silu_block_avx2);
trig_block_kernel!(
    sin_block,
    sin_one,
    f64::sin,
    sin_block_avx512,
    sin_block_avx2
);
trig_block_kernel!(
    cos_block,
    cos_one,
    f64::cos,
    cos_block_avx512,
    cos_block_avx2
);
trig_block_kernel!(
    tan_block,
    tan_one,
    f64::tan,
    tan_block_avx512,
    tan_block_avx2
);
block_kernel_param!(
    softplus_block,
    softplus_one,
    softplus_block_avx512,
    softplus_block_avx2
);
block_kernel_param!(
    exp_shifted_block,
    exp_shifted_one,
    exp_shifted_block_avx512,
    exp_shifted_block_avx2
);
block_kernel!(exp_block, exp_one, exp_block_avx512, exp_block_avx2);
block_kernel!(atan_block, atan_one, atan_block_avx512, atan_block_avx2);
block_kernel!(asin_block, asin_one, asin_block_avx512, asin_block_avx2);
block_kernel!(acos_block, acos_one, acos_block_avx512, acos_block_avx2);
block_kernel!(
    neg_log_sigmoid_block,
    neg_log_sigmoid_one,
    neg_log_sigmoid_block_avx512,
    neg_log_sigmoid_block_avx2
);
block_kernel!(expm1_block, expm1_one, expm1_block_avx512, expm1_block_avx2);
block_kernel!(sinh_block, sinh_one, sinh_block_avx512, sinh_block_avx2);
block_kernel!(cosh_block, cosh_one, cosh_block_avx512, cosh_block_avx2);
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
    silu_backward_block,
    silu_backward_one,
    silu_backward_block_avx512,
    silu_backward_block_avx2
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

    /// Write `sin(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn sin(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            sin_block,
            sin_block_avx512,
            sin_block_avx2
        )
    }

    /// Write `cos(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn cos(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            cos_block,
            cos_block_avx512,
            cos_block_avx2
        )
    }

    /// Write `tan(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn tan(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            tan_block,
            tan_block_avx512,
            tan_block_avx2
        )
    }

    /// Write `sigmoid(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn sigmoid(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            sigmoid_block,
            sigmoid_block_avx512,
            sigmoid_block_avx2
        )
    }

    /// Write `x * sigmoid(x)` for every element into `out`.
    #[inline]
    pub(crate) fn silu(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            silu_block,
            silu_block_avx512,
            silu_block_avx2
        )
    }

    /// Write the SiLU gradient for `input`, scaled by `grad`, into `out`.
    #[inline]
    pub(crate) fn silu_backward(self, input: &[f32], grad: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch2!(
            self,
            input,
            grad,
            out,
            silu_backward_block,
            silu_backward_block_avx512,
            silu_backward_block_avx2
        )
    }

    /// Write `exp(input[i] - shift)` into every element of `out`.
    #[inline]
    pub(crate) fn exp_shifted(self, input: &[f32], out: &mut [MaybeUninit<f32>], shift: f64) {
        dispatch_param!(
            self,
            input,
            out,
            shift,
            0.0,
            exp_shifted_block,
            exp_shifted_block_avx512,
            exp_shifted_block_avx2
        )
    }

    /// Write `-log(sigmoid(input[i]))` into every element of `out`.
    #[inline]
    pub(crate) fn neg_log_sigmoid(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            neg_log_sigmoid_block,
            neg_log_sigmoid_block_avx512,
            neg_log_sigmoid_block_avx2
        )
    }

    /// Write `asin(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn asin(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            asin_block,
            asin_block_avx512,
            asin_block_avx2
        )
    }

    /// Write `acos(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn acos(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            acos_block,
            acos_block_avx512,
            acos_block_avx2
        )
    }

    /// Write `atan(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn atan(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            atan_block,
            atan_block_avx512,
            atan_block_avx2
        )
    }

    /// Write `exp(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn exp(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            exp_block,
            exp_block_avx512,
            exp_block_avx2
        )
    }

    /// Write `log(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn log(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            log_block,
            log_block_avx512,
            log_block_avx2
        )
    }

    /// Write `log1p(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn log1p(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            log1p_block,
            log1p_block_avx512,
            log1p_block_avx2
        )
    }

    /// Write `softplus(input[i], beta, threshold)` into every element of `out`.
    #[inline]
    pub(crate) fn softplus(
        self,
        input: &[f32],
        out: &mut [MaybeUninit<f32>],
        beta: f64,
        threshold: f64,
    ) {
        dispatch_param!(
            self,
            input,
            out,
            beta,
            threshold,
            softplus_block,
            softplus_block_avx512,
            softplus_block_avx2
        )
    }

    /// Write `erfc(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn erfc(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            erfc_block,
            erfc_block_avx512,
            erfc_block_avx2
        )
    }

    /// Write `expm1(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn expm1(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            expm1_block,
            expm1_block_avx512,
            expm1_block_avx2
        )
    }

    /// Write `sinh(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn sinh(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            sinh_block,
            sinh_block_avx512,
            sinh_block_avx2
        )
    }

    /// Write `cosh(input[i])` into every element of `out`.
    #[inline]
    pub(crate) fn cosh(self, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        dispatch!(
            self,
            input,
            out,
            cosh_block,
            cosh_block_avx512,
            cosh_block_avx2
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
        Exp,
        Erf,
        GeluErf,
        GeluTanh,
        Expm1,
        Sinh,
        Cosh,
        Erfc,
        Log,
        Log1p,
        Sigmoid,
        Silu,
        Sin,
        Cos,
        Tan,
        Atan,
        Asin,
        Acos,
    }

    fn apply(op: Op, backend: Backend, input: &[f32], out: &mut [MaybeUninit<f32>]) {
        let k = F32Kernel(backend);
        match op {
            Op::Tanh => k.tanh(input, out),
            Op::Erf => k.erf(input, out),
            Op::GeluErf => k.gelu_erf(input, out),
            Op::GeluTanh => k.gelu_tanh(input, out),
            Op::Expm1 => k.expm1(input, out),
            Op::Sinh => k.sinh(input, out),
            Op::Cosh => k.cosh(input, out),
            Op::Erfc => k.erfc(input, out),
            Op::Exp => k.exp(input, out),
            Op::Log => k.log(input, out),
            Op::Log1p => k.log1p(input, out),
            Op::Sigmoid => k.sigmoid(input, out),
            Op::Silu => k.silu(input, out),
            Op::Sin => k.sin(input, out),
            Op::Cos => k.cos(input, out),
            Op::Tan => k.tan(input, out),
            Op::Atan => k.atan(input, out),
            Op::Asin => k.asin(input, out),
            Op::Acos => k.acos(input, out),
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

    /// `exp_shifted` is the one kernel that takes its argument in two
    /// pieces, so it needs a check of its own: every backend has to agree
    /// with the others, and all of them with `exp(x - shift)` evaluated in
    /// float64 and rounded once.
    ///
    /// The shifts are the ones `softmax` actually produces -- the maximum of
    /// the slice, so the largest argument is exactly zero -- including the
    /// case that motivates the shift at all, where `exp(x)` alone would
    /// overflow to infinity and `exp(x - max)` is an ordinary number.
    #[test]
    fn exp_shifted_matches_a_float64_reference_on_every_backend() {
        let inputs: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -37.5,
            88.0,
            89.0,
            1e-8,
            -1e-8,
            700.0,
            -700.0,
            1e20,
            -1e20,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        let shifts = [0.0f64, 1.0, -1.0, 88.0, 700.0, 1e20, f32::MAX as f64];

        for &shift in &shifts {
            let mut reference_run: Option<Vec<f32>> = None;
            for backend in available() {
                let mut out = vec![MaybeUninit::uninit(); inputs.len()];
                F32Kernel(backend).exp_shifted(&inputs, &mut out, shift);
                let got: Vec<f32> = out
                    .into_iter()
                    .map(|v| unsafe { v.assume_init() })
                    .collect();

                for (&x, &have) in inputs.iter().zip(got.iter()) {
                    let want = (x as f64 - shift).exp() as f32;
                    if want.is_nan() {
                        assert!(have.is_nan(), "{x} - {shift} on {backend:?}");
                        continue;
                    }
                    if !want.is_finite() {
                        // Overflow and underflow have to land on the same
                        // side, not merely close to it.
                        assert_eq!(have, want, "exp({x} - {shift}) on {backend:?}");
                        continue;
                    }
                    let tolerance = (want.abs() * 4.0 * f32::EPSILON).max(f32::MIN_POSITIVE);
                    assert!(
                        (have - want).abs() <= tolerance,
                        "exp({x} - {shift}) on {backend:?}: {have} against {want}"
                    );
                }

                match &reference_run {
                    None => reference_run = Some(got),
                    Some(first) => assert_eq!(
                        &got, first,
                        "backends disagree on exp(x - {shift}) at {backend:?}"
                    ),
                }
            }
        }
    }

    /// `-log(sigmoid(x))` is `softplus(-x)`, and the kernel has to agree with
    /// that on every backend and with a float64 evaluation of it.
    ///
    /// The reference is `softplus(-x)` written the cancellation-free way,
    /// `max(z, 0) + log1p(exp(-|z|))`. Spelling it as plain
    /// `log1p(exp(-x))` would make the *reference* the broken side: at
    /// `x = -800` that overflows to infinity, where the answer is 800.
    #[test]
    fn neg_log_sigmoid_matches_a_float64_reference_on_every_backend() {
        let inputs: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            15.0,
            -15.0,
            19.9,
            -19.9,
            20.1,
            -20.1,
            88.0,
            -88.0,
            800.0,
            -800.0,
            1e-8,
            -1e-8,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        let mut reference_run: Option<Vec<f32>> = None;
        for backend in available() {
            let mut out = vec![MaybeUninit::uninit(); inputs.len()];
            F32Kernel(backend).neg_log_sigmoid(&inputs, &mut out);
            let got: Vec<f32> = out
                .into_iter()
                .map(|v| unsafe { v.assume_init() })
                .collect();

            for (&x, &have) in inputs.iter().zip(got.iter()) {
                let z = -(x as f64);
                let want = (z.max(0.0) + (-z.abs()).exp().ln_1p()) as f32;
                if want.is_nan() {
                    assert!(have.is_nan(), "{x} on {backend:?}");
                    continue;
                }
                if !want.is_finite() {
                    assert_eq!(have, want, "-log(sigmoid({x})) on {backend:?}");
                    continue;
                }
                let tolerance = (want.abs() * 4.0 * f32::EPSILON).max(f32::MIN_POSITIVE);
                assert!(
                    (have - want).abs() <= tolerance,
                    "-log(sigmoid({x})) on {backend:?}: {have} against {want}"
                );
            }

            match &reference_run {
                None => reference_run = Some(got),
                Some(first) => {
                    assert_eq!(&got, first, "backends disagree at {backend:?}")
                }
            }
        }
    }

    /// The linear tail is not an optimisation: `exp(-x)` for a large negative
    /// `x` overflows even float64, and the answer there is `-x` itself.
    #[test]
    fn neg_log_sigmoid_converges_on_minus_x_instead_of_overflowing() {
        let mut out = vec![MaybeUninit::uninit(); 3];
        F32Kernel::select().neg_log_sigmoid(&[-800.0, -100.0, 800.0], &mut out);
        let got: Vec<f32> = out
            .into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect();
        assert_eq!(got[0], 800.0);
        assert_eq!(got[1], 100.0);
        assert_eq!(got[2], 0.0);
        assert!((800.0f64).exp().is_infinite(), "the premise of the tail");
    }

    /// The shift is what stops the exponential overflowing, so subtracting it
    /// has to happen before the exponential and not after.
    #[test]
    fn exp_shifted_survives_arguments_whose_exponential_would_not() {
        let mut out = vec![MaybeUninit::uninit(); 2];
        F32Kernel::select().exp_shifted(&[200.0, 100.0], &mut out, 200.0);
        let got: Vec<f32> = out
            .into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect();
        assert_eq!(got[0], 1.0);
        assert!(got[1] > 0.0 && got[1].is_finite(), "got {}", got[1]);
        assert!((200.0f32).exp().is_infinite(), "the premise of the shift");
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
            Op::Expm1 => xd.exp_m1() as f32,
            Op::Sinh => xd.sinh() as f32,
            Op::Cosh => xd.cosh() as f32,
            Op::Erfc => libm::erfc(xd) as f32,
            Op::Exp => xd.exp() as f32,
            Op::Log => xd.ln() as f32,
            Op::Log1p => xd.ln_1p() as f32,
            // Written the stable way: `1/(1 + exp(-x))` overflows for large
            // negative x, which is the bug these kernels fix.
            Op::Sigmoid => stable_logistic(xd) as f32,
            Op::Silu => (xd * stable_logistic(xd)) as f32,
            Op::Sin => xd.sin() as f32,
            Op::Cos => xd.cos() as f32,
            Op::Tan => xd.tan() as f32,
            Op::Atan => xd.atan() as f32,
            Op::Asin => xd.asin() as f32,
            Op::Acos => xd.acos() as f32,
        }
    }

    /// `1/(1 + exp(-x))`, branching on the sign so neither side overflows.
    fn stable_logistic(x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// The ops that come out bit-identical to the correctly-rounded float64
    /// value, rather than merely within an ulp of it. `cosh` is in this set
    /// even though it replaced glibc's `coshf` rather than a promoted scalar:
    /// it is exact anyway, which makes it an accuracy gain (`coshf` misrounds
    /// 22,628,918 of the 2^32 inputs).
    fn bit_exact_ops() -> [Op; 10] {
        [
            Op::Tanh,
            Op::Exp,
            Op::Expm1,
            Op::Sinh,
            Op::Cosh,
            Op::Log,
            Op::Sin,
            Op::Cos,
            Op::Tan,
            Op::Atan,
        ]
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
    fn bit_exact_ops_match_their_promoted_reference() {
        let xs = sample_inputs();
        for op in bit_exact_ops() {
            for backend in available() {
                for (&x, got) in xs.iter().zip(run(op, backend, &xs)) {
                    let want = reference(op, x);
                    if want.is_nan() {
                        assert!(got.is_nan(), "{op:?}/{backend:?}: {x:e}");
                        continue;
                    }
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{op:?}/{backend:?}: f({x:e}) gave {got:e}, want {want:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn erf_and_gelu_stay_within_one_ulp() {
        let xs = sample_inputs();
        for op in [
            Op::Erf,
            Op::Erfc,
            Op::GeluErf,
            Op::GeluTanh,
            Op::Log1p,
            Op::Sigmoid,
            Op::Silu,
            Op::Sin,
            Op::Cos,
            Op::Tan,
            Op::Atan,
            Op::Asin,
            Op::Acos,
        ] {
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
        for op in [
            Op::Tanh,
            Op::Erf,
            Op::GeluErf,
            Op::GeluTanh,
            Op::Expm1,
            Op::Sinh,
            Op::Cosh,
            Op::Erfc,
            Op::Log,
            Op::Log1p,
            Op::Sigmoid,
            Op::Silu,
            Op::Sin,
            Op::Cos,
            Op::Tan,
            Op::Atan,
            Op::Asin,
            Op::Acos,
        ] {
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
        // Only the genuinely odd kernels. GELU is not odd, `cosh` is even,
        // `expm1` is neither, and `acos` is a reflection rather than an
        // odd function -- `acos(-x)` is `pi - acos(x)`, checked below.
        for op in [Op::Tanh, Op::Erf, Op::Sinh, Op::Sin, Op::Tan, Op::Atan] {
            for backend in available() {
                for (p, n) in run(op, backend, &xs)
                    .into_iter()
                    .zip(run(op, backend, &neg))
                {
                    assert_eq!(p.to_bits(), (-n).to_bits(), "{op:?}/{backend:?}: {p:e}");
                }
            }
        }

        // `asin` is odd too, on the interval where it is defined at all.
        let unit: Vec<f32> = (0..=1000).map(|i| i as f32 * 1e-3).collect();
        let unit_neg: Vec<f32> = unit.iter().map(|v| -v).collect();
        for backend in available() {
            for (p, n) in
                run(Op::Asin, backend, &unit)
                    .into_iter()
                    .zip(run(Op::Asin, backend, &unit_neg))
            {
                assert_eq!(p.to_bits(), (-n).to_bits(), "asin/{backend:?}: {p:e}");
            }
        }

        // And `acos` reflects: the two sides must add to `pi` exactly as the
        // float64 reference does.
        for backend in available() {
            for (&x, (p, n)) in unit
                .iter()
                .zip(run(Op::Acos, backend, &unit).into_iter().zip(run(
                    Op::Acos,
                    backend,
                    &unit_neg,
                )))
            {
                let want = (std::f64::consts::PI - (x as f64).acos()) as f32;
                let apart = (n.to_bits() as i64 - want.to_bits() as i64).abs();
                assert!(
                    apart <= 1,
                    "acos/{backend:?} at {x:e}: {n:e} against {want:e}"
                );
                assert!(p.is_finite());
            }
        }
    }

    /// Signed zero survives, which the `copysign` at the end of each core is
    /// there for -- the arithmetic loses it (`+0.0 + -0.0` is `+0.0`).
    #[test]
    fn signed_zero_survives() {
        // `cosh(0)` is 1, not a signed zero, so it sits out; so does `acos`,
        // whose value at zero is `pi/2`.
        for op in [
            Op::Tanh,
            Op::Erf,
            Op::GeluErf,
            Op::GeluTanh,
            Op::Expm1,
            Op::Sinh,
            Op::Atan,
            Op::Asin,
        ] {
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
        for op in [
            Op::Tanh,
            Op::Erf,
            Op::GeluErf,
            Op::GeluTanh,
            Op::Expm1,
            Op::Sinh,
            Op::Cosh,
            Op::Erfc,
            Op::Log,
            Op::Log1p,
            Op::Sigmoid,
            Op::Silu,
            Op::Sin,
            Op::Cos,
            Op::Tan,
            Op::Atan,
            Op::Asin,
            Op::Acos,
        ] {
            for backend in available() {
                for len in 0..xs.len() {
                    for (&x, g) in xs[..len].iter().zip(run(op, backend, &xs[..len])) {
                        let want = reference(op, x);
                        // NaN payloads are not part of any kernel's contract.
                        if want.is_nan() {
                            assert!(g.is_nan(), "{op:?}/{backend:?}: len {len}, x {x:e}");
                            continue;
                        }
                        assert!(
                            ulps_apart(g, want) <= 1,
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
    #[ignore = "sweeps all 2^32 float32 inputs per op; takes a few minutes"]
    fn bit_exact_ops_match_their_promoted_reference_exhaustively() {
        for op in bit_exact_ops() {
            for backend in available() {
                let (worst, differing) = sweep(op, backend);
                println!("  {op:?}/{backend:?}: {differing} of 2^32 differ, worst {worst} ulp");
                assert_eq!(
                    differing, 0,
                    "{op:?}/{backend:?}: {differing} of 2^32 differ, worst {worst} ulp"
                );
            }
        }
    }

    /// `erf`'s claim is weaker than `tanh`'s and stated as a number: at most one
    /// ulp, on at most a couple of hundred of the 4.3 billion inputs. The bound
    /// is loose enough not to be a rounding-mode tripwire and tight enough that
    /// a real regression -- `libm::erff` misrounds 127.6 million -- fails it.
    #[test]
    #[ignore = "sweeps all 2^32 float32 inputs; takes ~1 minute"]
    fn erf_is_almost_always_correctly_rounded_exhaustively() {
        // Per-op bounds, each a few times the measured count: loose enough not
        // to be a rounding-mode tripwire, tight enough that a regression to the
        // routine being replaced (127.6M for `erff`, 20.0M for `erfcf`) fails.
        // `erfc` misrounds more than `erf` because below |x| = 2 it does have to
        // form `1 - erf`, which at x = 2 costs 7.7 bits.
        for (op, bound) in [(Op::Erf, 500u64), (Op::Erfc, 500_000), (Op::Log1p, 100)] {
            for backend in available() {
                let (worst, differing) = sweep(op, backend);
                println!("  {op:?}/{backend:?}: {differing} of 2^32 misrounded, worst {worst} ulp");
                assert!(worst <= 1, "{op:?}/{backend:?}: worst error {worst} ulp");
                assert!(
                    differing <= bound,
                    "{op:?}/{backend:?}: {differing} of 2^32 not correctly rounded"
                );
            }
        }
    }

    /// The gradient kernels, which take an extra operand and so go through
    /// their own dispatch. Checked against float64 references written the
    /// cancellation-free way, for the same reason the forward ones are.
    #[derive(Clone, Copy, Debug)]
    enum GradKernel {
        GeluErf,
        GeluTanh,
        Silu,
    }

    fn run2(op: GradKernel, backend: Backend, xs: &[f32], gs: &[f32]) -> Vec<f32> {
        let k = F32Kernel(backend);
        let mut out = vec![MaybeUninit::uninit(); xs.len()];
        match op {
            GradKernel::GeluErf => k.gelu_erf_backward(xs, gs, &mut out),
            GradKernel::GeluTanh => k.gelu_tanh_backward(xs, gs, &mut out),
            GradKernel::Silu => k.silu_backward(xs, gs, &mut out),
        }
        out.into_iter()
            .map(|v| unsafe { v.assume_init() })
            .collect()
    }

    fn reference2(op: GradKernel, x: f32, g: f32) -> f32 {
        let xd = x as f64;
        let local = match op {
            GradKernel::GeluErf => {
                let cdf = 0.5 * libm::erfc(-xd * std::f64::consts::FRAC_1_SQRT_2);
                let pdf = (-0.5 * xd * xd).exp() * 0.3989422804014327;
                cdf + xd * pdf
            }
            GradKernel::GeluTanh => {
                let x2 = xd * xd;
                let v = 0.7978845608028654 * (xd + 0.044715 * xd * x2);
                // sigmoid(2v) is 0.5*(1 + tanh(v)); sech^2(v) is 4*s*(1-s).
                let s = 1.0 / (1.0 + (-2.0 * v).exp());
                let sech2 = 4.0 * s * (1.0 - s);
                s + 0.5 * xd * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2)
            }
            GradKernel::Silu => {
                let s = stable_logistic(xd);
                // `1 - s` from the reciprocal, not by subtraction: past x = 40
                // the subtraction is exactly zero.
                let one_minus_s = if xd >= 0.0 {
                    let e = (-xd).exp();
                    e / (1.0 + e)
                } else {
                    1.0 / (1.0 + xd.exp())
                };
                s * (1.0 + xd * one_minus_s)
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
        for op in [GradKernel::GeluErf, GradKernel::GeluTanh, GradKernel::Silu] {
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
        // GELU only: SiLU's gradient is still 1.7e-16 at x = -40, because it
        // decays like x*exp(x) rather than exp(-x^2/2).
        for op in [GradKernel::GeluErf, GradKernel::GeluTanh] {
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

    /// SiLU's gradient decays like `x*exp(x)`, so it is still representable far
    /// out. It must neither plateau nor truncate early -- the scalar form it
    /// replaced went to exactly zero from about x = -89, where `exp(-x)`
    /// overflows float32.
    #[test]
    fn silu_backward_tail_decays_without_truncating() {
        let xs: Vec<f32> = vec![-20.0, -40.0, -60.0, -80.0, -95.0, -105.0, -200.0];
        let gs = vec![1.0f32; xs.len()];
        for backend in available() {
            let got = run2(GradKernel::Silu, backend, &xs, &gs);
            for (i, w) in got.windows(2).enumerate() {
                assert!(
                    w[1].abs() < w[0].abs() || w[1] == 0.0,
                    "{backend:?}: stopped decaying at {i}: {got:?}"
                );
            }
            // Still nonzero where the true value is representable ...
            assert!(got[4] != 0.0, "{backend:?}: truncated at x = -95: {got:?}");
            // ... and zero once it is not.
            assert_eq!(got[6], 0.0, "{backend:?}: should underflow at x = -200");
        }
    }

    #[test]
    fn gelu_backward_propagates_nan() {
        for op in [GradKernel::GeluErf, GradKernel::GeluTanh, GradKernel::Silu] {
            for backend in available() {
                let got = run2(op, backend, &[f32::NAN, 1.0, 1.0], &[1.0, f32::NAN, 1.0]);
                assert!(got[0].is_nan(), "{op:?}/{backend:?}: NaN input");
                assert!(got[1].is_nan(), "{op:?}/{backend:?}: NaN gradient");
                assert!(got[2].is_finite());
            }
        }
    }
}
