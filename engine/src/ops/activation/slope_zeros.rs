// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The slopes of `silu`, `gelu`, its tanh form and `mish` about their zeros.
//!
//! Each of these slopes is a sum of two terms that cancel where it crosses
//! zero, a little below the origin: `s (1 + x s(-x))` near -1.2785 for silu,
//! `Phi(x) + x phi(x)` near -0.7518 for gelu. Formed as float64 there a slope
//! is little but its terms' roundings -- at the inputs nearest the zero it had
//! no correct digit, a few hundredths away it was still twenty ulps out, and
//! in float32 now and then one. Across the span where the terms make an ulp of
//! themselves more than three of the slope, it is instead its Taylor series
//! about the zero, `d (c_1 + c_2 d + ...)` for `d = x - x0`, with `x0` held as a
//! pair so that `d` is exact but for one rounding: as many terms as keep the
//! series' tail below 2^-56 of the slope there, which evaluated in float64 is
//! within an ulp and a half of the exact slope. The coefficients are the
//! slopes' derivatives at their zeros, formed at ninety digits and rounded.

use crate::ops::arithmetic::extended::Pair;

/// How far either side of a zero the float32 kernels take the series, and
/// with how many of its terms: past 2^-6 the terms of a float64 slope make an
/// ulp of themselves at most seventy of the slope's, far below a float32 one,
/// and within it eight terms leave the tail under 2^-51 of the slope.
const SPAN_F32: f64 = 1.0 / 64.0;
const TERMS_F32: usize = 8;

/// A slope's zero and its series there.
pub(crate) struct SlopeZero<const N: usize> {
    /// The zero, as a pair.
    root: Pair,
    /// The span of `d = x - x0` taken by the series in float64.
    span: (f64, f64),
    /// The series' coefficients, from `d`'s first power.
    terms: [f64; N],
}

impl<const N: usize> SlopeZero<N> {
    /// `x - x0`, exact but for one rounding wherever `x` is within a factor
    /// of two of the zero, which every span is.
    #[inline(always)]
    fn offset(&self, x: f64) -> f64 {
        (x - self.root.0) - self.root.1
    }

    /// The slope at `x` from the series, where `x` lies in its span.
    #[inline]
    pub(crate) fn near(&self, x: f64) -> Option<f64> {
        let d = self.offset(x);
        if d.is_nan() || d < self.span.0 || d > self.span.1 {
            return None;
        }
        let mut p = 0.0;
        for &c in self.terms.iter().rev() {
            p = p * d + c;
        }
        Some(d * p)
    }

    /// For the float32 kernels: whether `x` lies within [`SPAN_F32`] of the
    /// zero, and the slope from the series' first terms -- formed whether or
    /// not it is wanted, so that a kernel's loop keeps no branch.
    #[inline(always)]
    pub(crate) fn near_f32<const FMA: bool>(&self, x: f64) -> (bool, f64) {
        let d = self.offset(x);
        let mut p = 0.0f64;
        for &c in self.terms[..TERMS_F32].iter().rev() {
            p = if FMA { p.mul_add(d, c) } else { p * d + c };
        }
        (d.abs() < SPAN_F32, d * p)
    }
}

/// `silu'(x) = s(x) (1 + x s(-x))`, zero near -1.2785.
pub(crate) const SILU: SlopeZero<34> = SlopeZero {
    root: Pair(-1.2784645427610737, -1.0946994183093437e-16),
    span: (-0.95, 0.55),
    terms: [
        0.2178117057198001,
        0.1466487969969469,
        0.018874814223782312,
        -0.015222655223188032,
        -0.006606589138356696,
        0.000126627410081122,
        0.0007985218818397998,
        0.00018570724361186496,
        -4.090534237428612e-05,
        -2.9733542213263917e-05,
        -2.942631888842464e-06,
        2.346029682463866e-06,
        8.599695028268575e-07,
        -3.051244750055421e-08,
        -9.266646309267441e-08,
        -1.8877622907727957e-08,
        4.619379769031606e-09,
        2.9177603026903803e-09,
        2.3648315053103244e-10,
        -2.251366201881081e-10,
        -7.50344609385287e-11,
        3.998547874723996e-12,
        8.04274075537398e-12,
        1.4842855116792256e-12,
        -4.1428341124873365e-13,
        -2.3496573817834053e-13,
        -1.5376365106809265e-14,
        1.8397842645887054e-14,
        5.662235988802348e-15,
        -4.0908332256286805e-16,
        -6.188638949724368e-16,
        -1.0399488154756224e-16,
        3.334275149650013e-17,
        1.721651999171468e-17,
    ],
};

/// `gelu'(x) = Phi(x) + x phi(x)`, zero near -0.7518.
pub(crate) const GELU_ERF: SlopeZero<21> = SlopeZero {
    root: Pair(-0.7517915246935645, 1.4956759177009883e-17),
    span: (-0.45, 0.31),
    terms: [
        0.4314939923140469,
        0.388284982990552,
        -0.018199676398671087,
        -0.1140082332972217,
        -0.014771522148244337,
        0.019421679838189067,
        0.004539228379125415,
        -0.002239538068073497,
        -0.0007448268386746817,
        0.00018633974623233514,
        8.615947861116571e-05,
        -1.121438018842664e-05,
        -7.74846130700372e-06,
        4.3284506257392273e-07,
        5.702764979242107e-07,
        -1.6044265494246427e-09,
        -3.5459866461783865e-08,
        -1.396942086426202e-09,
        1.90319840501314e-09,
        1.4506418938779426e-10,
        -8.950037692726255e-11,
    ],
};

/// The tanh form's slope, with the float64 constants its kernels use, zero near -0.7525.
pub(crate) const GELU_TANH: SlopeZero<23> = SlopeZero {
    root: Pair(-0.7524614220710162, -3.473691681308185e-17),
    span: (-0.45, 0.31),
    terms: [
        0.43040009102485854,
        0.38751844613578895,
        -0.015782853521848016,
        -0.11394448308095902,
        -0.01661932834305258,
        0.019682309459833125,
        0.005261059254921919,
        -0.0024227318458750974,
        -0.0009274420230205465,
        0.0002639276405268103,
        0.0001242522780263981,
        -3.495617169443604e-05,
        -1.5950896871645142e-05,
        5.91861171089399e-06,
        2.433551634429959e-06,
        -9.898647747433641e-07,
        -4.310248298802907e-07,
        1.4157556987729436e-07,
        7.52074090928993e-08,
        -1.6978217459780963e-08,
        -1.1961808479724423e-08,
        1.7637802707907462e-09,
        1.7449871378466231e-09,
    ],
};

/// `mish'(x) = tanh(sp) + x sech^2(sp) s(x)` for `sp = softplus(x)`, zero near -1.1924.
pub(crate) const MISH: SlopeZero<40> = SlopeZero {
    root: Pair(-1.1924312145154952, -4.8484829848031044e-17),
    span: (-0.95, 0.52),
    terms: [
        0.2669479140495345,
        0.20473126408010586,
        0.04190782104360987,
        -0.020271822716684245,
        -0.01582112656173338,
        -0.0033606849270232685,
        0.0010924055409445854,
        0.0009898181021289196,
        0.00025412936386191073,
        -4.1961496031696126e-05,
        -5.582891567360688e-05,
        -1.72992708103044e-05,
        9.427361327376651e-07,
        2.9088199364912223e-06,
        1.071676702702074e-06,
        2.6879417855557728e-08,
        -1.4117255919499382e-07,
        -6.174424363104907e-08,
        -5.460186173028381e-09,
        6.3895954615089855e-09,
        3.3604012383661332e-09,
        4.817646300218697e-10,
        -2.6751871594311224e-10,
        -1.7442209713382678e-10,
        -3.3904729895707114e-11,
        1.0101727316036154e-11,
        8.681382594457511e-12,
        2.124385967466607e-12,
        -3.20344479568135e-13,
        -4.153673469447499e-13,
        -1.2348725029307754e-13,
        6.337359322280651e-15,
        1.9103418630556258e-14,
        6.79383226720695e-15,
        1.6183924412509285e-16,
        -8.421603581894615e-16,
        -3.577018575431464e-16,
        -3.059285498528877e-17,
        3.534620906867835e-17,
        1.8140677609631626e-17,
    ],
};
