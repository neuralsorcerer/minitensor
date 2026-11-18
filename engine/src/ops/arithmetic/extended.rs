// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `exp`, `expm1` and `log1p` carried past float64, as pairs of floats whose
//! sum is the value to about 2^-100 of it.
//!
//! They are for the few results that a float64 `exp` or `log1p` cannot reach:
//! sums that cancel what float64 holds of their terms. `logaddexp(a, b)` near
//! where `e^a + e^b` is one is a number near `-log1p(e^(b - a))` plus `a`, and
//! float64 terms leave it with their error, an ulp of a number many times
//! larger. The pairs carry the terms far enough that the sum rounds as the
//! exact one does. They cost some hundreds of operations, so callers take
//! them only where the float64 result has cancelled.
//!
//! Each is generic over `FMA`, whether the code it is compiled into may use a
//! fused multiply-add for a product's exact error; see [`fused`], which runs a
//! caller's body on the faster form where the CPU has it.

use super::rounded_once::{product_error, two_sum};

/// A value as an unevaluated sum of two floats, the second at most half an
/// ulp of the first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pair(pub(crate) f64, pub(crate) f64);

/// `log 2` and `log2(e)` as pairs.
pub(crate) const LN_2: Pair = Pair(std::f64::consts::LN_2, 2.319_046_813_846_299_6e-17);
pub(crate) const LOG2_E: Pair = Pair(std::f64::consts::LOG2_E, 2.035_527_374_093_103_3e-17);

/// `log 2 / 64` as a pair: the step of [`exp_parts`]' reduction.
const LN_2_64: Pair = Pair(0.010_830_424_696_249_145, 3.623_510_646_634_843e-19);

/// `1 / k!` for `k` from 2 to 5, as pairs: the terms of `expm1(r)` whose
/// float64 rounding would show at `|r| = log 2 / 128`. The rest, to the
/// eleventh, are float64 in [`exp_parts`].
const SERIES: [Pair; 4] = [
    Pair(0.5, 0.0),
    Pair(0.166_666_666_666_666_66, 9.251_858_538_542_97e-18),
    Pair(0.041_666_666_666_666_664, 2.312_964_634_635_742_7e-18),
    Pair(0.008_333_333_333_333_333, 1.156_482_317_317_871_4e-19),
];

/// `2^(j/64)` for `j` from 0 to 63, as pairs.
#[rustfmt::skip]
const POWERS: [Pair; 64] = [
    Pair(1.0, 0.0),
    Pair(1.0108892860517005, -1.5234778603368577e-17),
    Pair(1.0218971486541166, 5.109225028973444e-17),
    Pair(1.0330248790212284, 7.600838874027088e-18),
    Pair(1.0442737824274138, 8.551889705537965e-17),
    Pair(1.0556451783605572, 1.759325738772092e-18),
    Pair(1.0671404006768237, -7.899853966841582e-17),
    Pair(1.0787607977571199, -6.656660436056593e-17),
    Pair(1.0905077326652577, -3.046782079812471e-17),
    Pair(1.102382583307841, 5.2660368715706944e-17),
    Pair(1.1143867425958924, 1.0410278456845571e-16),
    Pair(1.1265216186082418, 5.165856758795457e-17),
    Pair(1.1387886347566916, 8.912812676025408e-17),
    Pair(1.1511892299529827, 3.250710218863827e-17),
    Pair(1.1637248587775775, 3.8292048369240935e-17),
    Pair(1.1763969916502812, 5.554203254218079e-17),
    Pair(1.189207115002721, 3.982015231465646e-17),
    Pair(1.202156731452703, 6.644981499252301e-17),
    Pair(1.215247359980469, -7.712630692681488e-17),
    Pair(1.22848053610687, -1.89878163130253e-17),
    Pair(1.241857812073484, 4.658027591836937e-17),
    Pair(1.255380757024691, -6.7113898212968784e-18),
    Pair(1.2690509571917332, 2.667932131342186e-18),
    Pair(1.2828700160787783, 1.713594918243561e-17),
    Pair(1.2968395546510096, 2.5382502794888315e-17),
    Pair(1.3109612115247644, -7.181536135519454e-17),
    Pair(1.3252366431597413, -2.8587312100388614e-17),
    Pair(1.339667524053303, 8.927282594831732e-17),
    Pair(1.3542555469368927, 7.70094837980299e-17),
    Pair(1.3690024229745905, 9.593797919118849e-17),
    Pair(1.383909881963832, -6.770511658794786e-17),
    Pair(1.3989796725383112, -9.614213209051323e-17),
    Pair(std::f64::consts::SQRT_2, -9.667293313452913e-17),
    Pair(1.42961333839197, -1.2031642489053655e-17),
    Pair(1.4451808069770467, -3.0237581349939873e-17),
    Pair(1.460917794180647, -5.600377186075216e-17),
    Pair(1.4768261459394993, -3.483994556892796e-17),
    Pair(1.4929077282912648, 1.4192920154284036e-17),
    Pair(1.5091644275934228, -1.016455327754295e-16),
    Pair(1.5255981507445384, -1.1024941712342561e-16),
    Pair(1.5422108254079407, 7.949834809697621e-17),
    Pair(1.559004400237837, 3.7812070533575275e-17),
    Pair(1.5759808451078865, -1.0136916471278304e-17),
    Pair(1.593142151342267, -1.0094406542311964e-16),
    Pair(1.6104903319492543, 2.4707192569797888e-17),
    Pair(1.6280274218573478, -6.712955084707084e-17),
    Pair(1.645755478153965, -1.0125679913674773e-16),
    Pair(1.6636765803267364, 5.8909926967131e-17),
    Pair(1.681792830507429, 8.199010020581497e-17),
    Pair(1.7001063537185235, -8.0237193703977e-18),
    Pair(1.718619298122478, -1.851380418263111e-17),
    Pair(1.7373338352737062, 3.164389299292957e-17),
    Pair(1.7562521603732995, 2.960140695448873e-17),
    Pair(1.7753764925265212, 6.429731796556572e-17),
    Pair(1.7947090750031072, 1.8227458427912087e-17),
    Pair(1.8142521755003989, -9.969531538920349e-17),
    Pair(1.8340080864093424, 3.283107224245627e-17),
    Pair(1.8539791250833855, 9.761887490727594e-17),
    Pair(1.8741676341103, -6.122763413004143e-17),
    Pair(1.8945759815869656, 3.4034035352165297e-17),
    Pair(1.9152065613971474, -1.0619946056195963e-16),
    Pair(1.9360617934922943, 1.0332385960676326e-16),
    Pair(1.9571441241754002, 8.960767791036668e-17),
    Pair(1.978456026387951, 4.0388753109278167e-17),
];

/// The exact error of `p`, the float64 nearest `x * y`: one fused
/// multiply-add where `FMA` says the code runs where there is one, and
/// otherwise both operands split into halves whose products are exact.
#[inline(always)]
fn error_of<const FMA: bool>(x: f64, y: f64, p: f64) -> f64 {
    if FMA {
        x.mul_add(y, -p)
    } else {
        product_error(x, y, p)
    }
}

/// `a + b` where `|a| >= |b|`, and its exact rounding error.
#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}

impl Pair {
    #[inline(always)]
    fn normal(hi: f64, lo: f64) -> Self {
        let (hi, lo) = fast_two_sum(hi, lo);
        Pair(hi, lo)
    }

    /// The sum of two pairs: the leading parts' exactly, the trailing parts'
    /// rounded beside its error. Where the pairs cancel, their leading parts
    /// do so exactly, and what rounds is the trailing parts' sum, at 2^-106 of
    /// the operands -- below anything the callers here keep.
    #[inline(always)]
    pub(crate) fn add(self, other: Pair) -> Pair {
        let (s, e) = two_sum(self.0, other.0);
        Pair::normal(s, e + (self.1 + other.1))
    }

    pub(crate) fn neg(self) -> Pair {
        Pair(-self.0, -self.1)
    }

    #[inline(always)]
    pub(crate) fn mul<const FMA: bool>(self, other: Pair) -> Pair {
        let p = self.0 * other.0;
        let e = error_of::<FMA>(self.0, other.0, p) + (self.0 * other.1 + self.1 * other.0);
        Pair::normal(p, e)
    }

    /// Times a power of two, exactly while both parts stay normal.
    fn scaled(self, power: f64) -> Pair {
        Pair(self.0 * power, self.1 * power)
    }
}

/// `exp(x)` for `x = hi + lo` as `2^n 2^(j/64) (1 + e)`: `64 n + j` the
/// nearest whole number to `64 x / log 2`, and `e` the pair
/// `expm1(x - (64 n + j) log 2 / 64)`.
///
/// The reduced argument, at most `log 2 / 128` in size, is formed as a pair
/// -- the leading product of the step exact, and the rest a small float --
/// and its series is pairs to the fifth term and float64 past it, to the
/// eleventh, where the next is below 2^-110 of the whole. `x` is finite and
/// at most 708 in size, so `2^n` is a normal float.
#[inline(always)]
fn exp_parts<const FMA: bool>(x: Pair) -> (f64, usize, Pair) {
    let steps = (x.0 * (64.0 * LOG2_E.0)).round();
    let lead = steps * LN_2_64.0;
    let r = x
        .add(Pair(-lead, -error_of::<FMA>(steps, LN_2_64.0, lead)))
        .add(Pair(-steps * LN_2_64.1, 0.0));
    let n = (steps / 64.0).floor();
    let j = (steps - 64.0 * n) as usize;
    let s = r.0;
    let tail = (((((s * (1.0 / 39_916_800.0) + 1.0 / 3_628_800.0) * s + 1.0 / 362_880.0) * s
        + 1.0 / 40_320.0)
        * s
        + 1.0 / 5_040.0)
        * s
        + 1.0 / 720.0)
        * s;
    let mut q = Pair(tail, 0.0);
    for &c in SERIES.iter().rev() {
        q = q.add(c).mul::<FMA>(r);
    }
    // `q` is now `expm1(r) / r - 1`.
    (n, j, q.add(Pair(1.0, 0.0)).mul::<FMA>(r))
}

/// `exp(x)` as a pair, for `x` from -708 to 708.
#[inline(always)]
pub(crate) fn exp<const FMA: bool>(x: Pair) -> Pair {
    let (n, j, e) = exp_parts::<FMA>(x);
    let power = POWERS[j];
    power.add(power.mul::<FMA>(e)).scaled(2f64.powi(n as i32))
}

/// `exp(x) - 1` as a pair, for `x` from -708 to 708: the reduced `e` itself
/// where the reduction took nothing out, which keeps a small result's digits.
#[inline(always)]
pub(crate) fn expm1<const FMA: bool>(x: Pair) -> Pair {
    let (n, j, e) = exp_parts::<FMA>(x);
    if n == 0.0 && j == 0 {
        return e;
    }
    let power = POWERS[j];
    power
        .add(power.mul::<FMA>(e))
        .scaled(2f64.powi(n as i32))
        .add(Pair(-1.0, 0.0))
}

/// `log(1 + w)` as a pair, for `w` above -1 and below 2^1000.
///
/// One Newton step from the float64 `log1p` of the leading part, on
/// `expm1(l) = w`: the step halves the digits the float64 one missed, squared,
/// and `w - expm1(l)` is formed as pairs where both are near `w`, so a small
/// `w` keeps its relative accuracy.
#[inline(always)]
pub(crate) fn ln_1p<const FMA: bool>(w: Pair) -> Pair {
    let l = w.0.ln_1p();
    if l == 0.0 || !l.is_finite() {
        return Pair(l + w.1, 0.0);
    }
    let m = expm1::<FMA>(Pair(l, 0.0));
    let gap = w.add(m.neg());
    Pair::normal(l, gap.0 / (1.0 + m.0))
}

/// Run `body`'s float64 arithmetic with `FMA` as the CPU allows: compiled with
/// the fused multiply-add where it has one -- a third less work for every
/// product of pairs -- and without it otherwise. Written
/// `fused!(body(name: Type, ...) -> Return)`, each `name` a local passed on.
macro_rules! fused {
    ($body:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[target_feature(enable = "fma")]
            fn with_fma($($arg: $ty),*) -> $ret {
                $body::<true>($($arg),*)
            }
            if crate::ops::simd::simd_capabilities().avx2_fma {
                // SAFETY: fma was detected on this CPU.
                unsafe { with_fma($($arg),*) }
            } else {
                $body::<false>($($arg),*)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $body::<{ cfg!(target_arch = "aarch64") }>($($arg),*)
        }
    }};
}
pub(crate) use fused;

#[cfg(test)]
mod tests {
    use super::*;

    fn value(p: Pair) -> f64 {
        p.0 + p.1
    }

    #[test]
    fn exp_of_whole_logs_of_two_is_the_power() {
        for n in [-20, -1, 0, 1, 7, 30] {
            let got = exp::<false>(Pair(n as f64, 0.0).mul::<false>(LN_2));
            // `n log 2` is a pair, so the result is a power of two to
            // within its own last parts.
            let want = 2f64.powi(n);
            assert!((value(got) - want).abs() <= want * 1e-30, "{n}: {got:?}");
        }
    }

    #[test]
    fn expm1_and_ln_1p_invert_each_other() {
        for &w in &[1e-300, 1e-20, 1e-9, 0.001, 0.3, 0.999, 1.0, 5.0, 1e10] {
            let l = ln_1p::<false>(Pair(w, 0.0));
            let back = expm1::<false>(l);
            let gap = back.add(Pair(-w, 0.0));
            assert!(value(gap).abs() <= w * 1e-29, "{w}: {back:?}");
        }
    }

    #[test]
    fn ln_1p_of_one_is_log_two() {
        let got = ln_1p::<false>(Pair(1.0, 0.0));
        assert_eq!(got.0, LN_2.0);
        assert!((got.1 - LN_2.1).abs() <= 1e-31, "{got:?}");
    }
}
