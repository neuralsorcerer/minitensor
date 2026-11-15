// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `digamma` and `lgamma` left of zero, their reflection formulas carried
//! past float64's precision.
//!
//! Left of zero both are reflected, `digamma(x) = digamma(-x) - 1/x -
//! pi cot(pi x)` and `lgamma(x) = ln pi - ln |x sin(pi x)| - lgamma(-x)`, and
//! between each pair of negative integers each crosses zero, where its terms
//! -- of the size of `ln |x|` -- cancel. Taken in float64 the answer there was
//! what the terms rounded: `lgamma` came out 16% low at the float nearest its
//! zero at -2.457 and `digamma` 1.8e15 ulps out next to its zero at -1.573,
//! and across (-3, -2) half of `lgamma`'s answers missed by an ulp or more.
//!
//! Here the terms are carried in pairs of float64s, about 106 bits, with a
//! bound on what that rounds; where the bound places the float64 nearest the
//! answer, that is the answer. Next to a zero, where the answer is as small
//! as `1e-16` against terms of size one, it does not, and the terms are taken
//! again in triples, about 159 bits.

use crate::ops::arithmetic::exact_sum::Output;
use crate::ops::arithmetic::rounded_once::{product_error, two_sum};

// The constants below to three float64s each, the second and third what the
// first and the first two leave of the exact value: `pi`, `ln 2`, `ln pi`,
// `ln(2 pi) / 2`; `1 / (2k + 1)`; `(-1)^k / (2k + 1)!` and
// `(-1)^k / (2k)!`; and, for `k` from one, `B_2k / 2k` and
// `B_2k / (2k (2k - 1))`, the coefficients of the asymptotic expansions of
// `digamma` and of `lgamma`.
const PI: [f64; 3] = [
    std::f64::consts::PI,
    1.2246467991473532e-16,
    -2.9947698097183397e-33,
];
const LN_2: [f64; 3] = [
    std::f64::consts::LN_2,
    2.3190468138462996e-17,
    5.707708438416212e-34,
];
const LN_PI: [f64; 3] = [
    1.1447298858494002,
    1.0265951162707826e-17,
    -1.3722612652165766e-34,
];
const HALF_LN_TWO_PI: [f64; 3] = [
    0.9189385332046728,
    -3.8782941580672414e-17,
    -1.323971596849807e-33,
];
const ODD: [[f64; 3]; 21] = [
    [1.0, 0.0, 0.0],
    [
        0.3333333333333333,
        1.850371707708594e-17,
        1.0271626370065257e-33,
    ],
    [0.2, -1.1102230246251566e-17, 6.162975822039155e-34],
    [
        0.14285714285714285,
        7.93016446160826e-18,
        4.4021255871708246e-34,
    ],
    [
        0.1111111111111111,
        6.1679056923619804e-18,
        3.423875456688419e-34,
    ],
    [
        0.09090909090909091,
        -2.523234146875356e-18,
        7.003381615953585e-35,
    ],
    [
        0.07692307692307693,
        -4.270088556250602e-18,
        2.370375316168906e-34,
    ],
    [
        0.06666666666666667,
        9.251858538542971e-19,
        1.2839532962581572e-35,
    ],
    [
        0.058823529411764705,
        8.163404592832033e-19,
        1.1328999672866093e-35,
    ],
    [
        0.05263157894736842,
        2.921639538487254e-18,
        1.6218357426418827e-34,
    ],
    [
        0.047619047619047616,
        2.64338815386942e-18,
        1.4673751957236082e-34,
    ],
    [
        0.043478260869565216,
        1.206764157201257e-18,
        3.349443381543019e-35,
    ],
    [0.04, -8.326672684688674e-19, -3.0814879110195774e-35],
    [
        0.037037037037037035,
        2.05596856412066e-18,
        1.1412918188961397e-34,
    ],
    [
        0.034482758620689655,
        4.785444071660157e-19,
        6.64113773926633e-36,
    ],
    [
        0.03225806451612903,
        8.953411488912552e-19,
        2.48507089598353e-35,
    ],
    [
        0.030303030303030304,
        -8.410780489584519e-19,
        2.334460538651195e-35,
    ],
    [
        0.02857142857142857,
        8.921435019309293e-19,
        8.804251174341649e-35,
    ],
    [
        0.02702702702702703,
        -1.50030138462859e-18,
        8.328345705458318e-35,
    ],
    [
        0.02564102564102564,
        8.896017825522087e-19,
        7.901251053896352e-35,
    ],
    [
        0.024390243902439025,
        -8.46206573647223e-19,
        -1.8789560433046204e-35,
    ],
];
const SINE: [[f64; 3]; 20] = [
    [1.0, 0.0, 0.0],
    [
        -0.16666666666666666,
        -9.25185853854297e-18,
        -5.135813185032629e-34,
    ],
    [
        0.008333333333333333,
        1.1564823173178714e-19,
        1.6049416203226965e-36,
    ],
    [
        -0.0001984126984126984,
        -1.7209558293420705e-22,
        -1.4926912391394127e-40,
    ],
    [
        2.7557319223985893e-06,
        -1.858393274046472e-22,
        8.491754604881993e-39,
    ],
    [
        -2.505210838544172e-08,
        1.448814070935912e-24,
        -2.0426735146714455e-41,
    ],
    [
        1.6059043836821613e-10,
        1.2585294588752098e-26,
        -5.31334602762985e-43,
    ],
    [
        -7.647163731819816e-13,
        -7.03872877733453e-30,
        7.827539277162583e-48,
    ],
    [
        2.8114572543455206e-15,
        1.6508842730861433e-31,
        -2.877771793074479e-50,
    ],
    [
        -8.22063524662433e-18,
        -2.2141894119604265e-34,
        1.508914023774199e-50,
    ],
    [
        1.9572941063391263e-20,
        -1.3643503830087908e-36,
        1.3392348251125064e-53,
    ],
    [
        -3.868170170630684e-23,
        8.843177655482344e-40,
        -3.8718157106173247e-56,
    ],
    [
        6.446950284384474e-26,
        -1.9330404233703465e-42,
        -1.5213023807039144e-58,
    ],
    [
        -9.183689863795546e-29,
        -1.4303150396787322e-45,
        8.551226774650505e-62,
    ],
    [
        1.1309962886447716e-31,
        1.0498015412959506e-47,
        -4.346150929397795e-64,
    ],
    [
        -1.216125041553518e-34,
        -5.586290567888806e-51,
        -6.615948578082792e-68,
    ],
    [
        1.151633562077195e-37,
        -6.09957445788454e-54,
        -5.3447496196594105e-70,
    ],
    [
        -9.67759295863189e-41,
        -3.202295548645562e-57,
        -6.547507205018101e-74,
    ],
    [
        7.265460179153071e-44,
        -4.364097149354446e-61,
        2.5503250121018375e-77,
    ],
    [
        -4.902469756513544e-47,
        1.213019100517928e-63,
        4.4707180011376586e-80,
    ],
];
const COSINE: [[f64; 3]; 20] = [
    [1.0, 0.0, 0.0],
    [-0.5, 0.0, 0.0],
    [
        0.041666666666666664,
        2.3129646346357427e-18,
        1.2839532962581572e-34,
    ],
    [
        -0.001388888888888889,
        5.300543954373577e-20,
        1.7386867553495878e-36,
    ],
    [
        2.48015873015873e-05,
        2.1511947866775882e-23,
        1.865864048924266e-41,
    ],
    [
        -2.755731922398589e-07,
        -2.3767714622250297e-23,
        3.263188903340883e-40,
    ],
    [
        2.08767569878681e-09,
        -1.20734505911326e-25,
        1.702227928892871e-42,
    ],
    [
        -1.1470745597729725e-11,
        -2.0655512752830745e-28,
        -6.889079232466646e-45,
    ],
    [
        4.779477332387385e-14,
        4.399205485834081e-31,
        -4.892212048226615e-49,
    ],
    [
        -1.5619206968586225e-16,
        -1.1910679660273754e-32,
        4.577506059629983e-49,
    ],
    [
        4.110317623312165e-19,
        1.4412973378659527e-36,
        -5.285627548789812e-53,
    ],
    [
        -8.896791392450574e-22,
        7.911402614872376e-38,
        3.1877976790570933e-54,
    ],
    [
        1.6117375710961184e-24,
        -3.6846573564509766e-41,
        1.613256546090552e-57,
    ],
    [
        -2.4795962632247976e-27,
        1.2953730964765229e-43,
        -6.403390159849962e-60,
    ],
    [
        3.279889237069838e-30,
        1.5117542744029879e-46,
        8.058517719519716e-63,
    ],
    [
        -3.7699876288159054e-33,
        -2.5870347832750324e-49,
        -3.23789002742564e-66,
    ],
    [
        3.8003907548547434e-36,
        1.7457158024652518e-52,
        2.0674839306508725e-69,
    ],
    [
        -3.387157535521162e-39,
        -5.09056148151085e-56,
        -3.989567349036344e-72,
    ],
    [
        2.6882202662866363e-42,
        5.355061165943334e-59,
        -1.1290601987449868e-75,
    ],
    [
        -1.911963205040282e-45,
        2.7860822176883126e-62,
        -2.0347437224101328e-78,
    ],
];
const DIGAMMA: [[f64; 3]; 20] = [
    [
        0.08333333333333333,
        4.625929269271485e-18,
        2.5679065925163143e-34,
    ],
    [
        -0.008333333333333333,
        -1.1564823173178714e-19,
        -1.6049416203226965e-36,
    ],
    [
        0.003968253968253968,
        2.20282346155785e-19,
        1.2228126631030068e-35,
    ],
    [
        -0.004166666666666667,
        -5.782411586589357e-20,
        -8.024708101613483e-37,
    ],
    [
        0.007575757575757576,
        -2.1026951223961299e-19,
        5.8361513466279876e-36,
    ],
    [
        -0.021092796092796094,
        1.3911677399530732e-18,
        -6.499719616955214e-35,
    ],
    [
        0.08333333333333333,
        4.625929269271485e-18,
        2.5679065925163143e-34,
    ],
    [
        -0.4432598039215686,
        -2.0462934179365632e-17,
        1.2567636970432786e-33,
    ],
    [
        3.0539543302701198,
        -1.0882720820608607e-17,
        -6.041123612803627e-34,
    ],
    [
        -26.456212121212122,
        7.449932926454383e-16,
        3.4542545698314e-32,
    ],
    [
        281.46014492753625,
        -1.647635329298783e-14,
        5.945038706013422e-31,
    ],
    [
        -3607.5105463980462,
        -1.5347029033579816e-13,
        1.0122077510117911e-29,
    ],
    [
        54827.583333333336,
        -2.4253192047278085e-12,
        -1.3463226115771934e-28,
    ],
    [
        -974936.8238505747,
        -4.2284185858978625e-11,
        2.6443633088081705e-27,
    ],
    [
        20052695.79668808,
        -8.878409127275055e-10,
        -7.700792371205397e-28,
    ],
    [
        -472384867.7216299,
        1.5660828235102635e-08,
        -6.09842961411644e-25,
    ],
    [
        12635724795.916666,
        6.357828776041666e-07,
        3.529303946892918e-23,
    ],
    [
        -380879311252.4537,
        -1.379912397079338e-05,
        7.244909722446504e-22,
    ],
    [
        12850850499305.084,
        -0.0006510416666666666,
        -3.614007241618348e-20,
    ],
    [
        -482414483548501.7,
        -0.016215816703621582,
        4.1961589003524716e-19,
    ],
];
const LGAMMA: [[f64; 3]; 20] = [
    [
        0.08333333333333333,
        4.625929269271485e-18,
        2.5679065925163143e-34,
    ],
    [
        -0.002777777777777778,
        1.0601087908747154e-19,
        3.4773735106991755e-36,
    ],
    [
        0.0007936507936507937,
        6.883823317368282e-22,
        5.970764956557651e-40,
    ],
    [
        -0.0005952380952380953,
        5.36938218754726e-20,
        -1.8342189946545105e-36,
    ],
    [
        0.0008417508417508417,
        3.6870174889237694e-20,
        -6.889900895324708e-37,
    ],
    [
        -0.0019175269175269176,
        1.0675702776872475e-19,
        6.568342495426554e-37,
    ],
    [
        0.00641025641025641,
        2.2240044563805217e-19,
        1.975312763474088e-35,
    ],
    [
        -0.029550653594771242,
        4.861760957508855e-19,
        1.316681517535326e-35,
    ],
    [
        0.17964437236883057,
        -6.401600482710946e-19,
        9.779977439678332e-36,
    ],
    [
        -1.3924322169059011,
        1.5837056989230303e-17,
        5.2056012685038854e-34,
    ],
    [
        13.402864044168393,
        -6.154114101993966e-16,
        1.3610436598016077e-34,
    ],
    [
        -156.84828462600203,
        9.391823141715389e-15,
        1.6570392471086158e-31,
    ],
    [
        2193.1033333333335,
        -1.3339255626002948e-13,
        6.731613057885968e-31,
    ],
    [
        -36108.77125372499,
        5.897583353514365e-13,
        7.049709715793733e-31,
    ],
    [
        691472.268851313,
        2.5585296305158e-11,
        -1.2521722821640843e-27,
    ],
    [
        -15238221.539407415,
        -8.76774522490625e-10,
        -1.9672353593923997e-26,
    ],
    [
        382900751.39141417,
        -2.4082684757733585e-08,
        -4.344787055834085e-25,
    ],
    [
        -10882266035.784391,
        3.141830930219749e-07,
        -2.013934646419947e-23,
    ],
    [
        347320283765.00226,
        -6.048528997747748e-06,
        5.341649216919011e-23,
    ],
    [
        -12369602142269.275,
        0.0009363732896507286,
        3.299942635958079e-20,
    ],
];

/// A bound on what the reflections below round in pairs, in proportion to the
/// magnitude of the terms they take the difference of: `2^-96`, against the
/// `2^-103.7` the most of any of 5000 arguments came to, the floats either
/// side of fifty zeros among them.
const PAIR_ERROR: f64 = 1.262_177_448_353_619e-29;

/// `a + b` and its rounding error, for `|a|` at least `|b|`.
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}

/// `a · b` and its rounding error.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, product_error(a, b, product))
}

/// A float wider than float64: the unevaluated sum of two or three of them,
/// each within about half an ulp of the one before.
trait Wide: Copy {
    const ZERO: Self;
    const ONE: Self;
    /// Where the asymptotic expansions hold to this width.
    const EXPANSION_MIN: f64;
    /// Of each series, how many terms are taken at this width and how many
    /// in all, those between in float64: the logarithm's, the sine's and the
    /// cosine's, and the expansions'.
    const LOG_TERMS: (usize, usize);
    const SINE_TERMS: (usize, usize);
    const EXPANSION_TERMS: (usize, usize);

    fn from_table(value: [f64; 3]) -> Self;
    fn of(value: f64) -> Self;
    /// The leading float64.
    fn head(self) -> f64;
    /// The float64 nearest the whole.
    fn value(self) -> f64;
    fn negated(self) -> Self;
    fn plus(self, other: Self) -> Self;
    fn plus_f(self, value: f64) -> Self;
    fn times(self, other: Self) -> Self;
    fn times_f(self, value: f64) -> Self;
    fn over(self, other: Self) -> Self;
    /// Times a power of two, which is exact.
    fn scaled(self, power: f64) -> Self;

    fn minus(self, other: Self) -> Self {
        self.plus(other.negated())
    }

    fn reciprocal(self) -> Self {
        Self::ONE.over(self)
    }

    fn magnitude(self) -> f64 {
        self.head().abs()
    }
}

/// Two float64s, about 106 bits.
#[derive(Clone, Copy, Debug)]
struct Pair {
    hi: f64,
    lo: f64,
}

impl Wide for Pair {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    const EXPANSION_MIN: f64 = 20.0;
    const LOG_TERMS: (usize, usize) = (11, 21);
    const SINE_TERMS: (usize, usize) = (8, 15);
    const EXPANSION_TERMS: (usize, usize) = (5, 15);

    fn from_table(value: [f64; 3]) -> Self {
        Self {
            hi: value[0],
            lo: value[1],
        }
    }

    fn of(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    fn head(self) -> f64 {
        self.hi
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }

    fn negated(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn plus(self, other: Self) -> Self {
        let (sum, error) = two_sum(self.hi, other.hi);
        let (low, low_error) = two_sum(self.lo, other.lo);
        let (sum, error) = quick_two_sum(sum, error + low);
        let (hi, lo) = quick_two_sum(sum, error + low_error);
        Self { hi, lo }
    }

    fn plus_f(self, value: f64) -> Self {
        let (sum, error) = two_sum(self.hi, value);
        let (hi, lo) = quick_two_sum(sum, error + self.lo);
        Self { hi, lo }
    }

    fn times(self, other: Self) -> Self {
        let (product, error) = two_prod(self.hi, other.hi);
        let (hi, lo) = quick_two_sum(product, error + (self.hi * other.lo + self.lo * other.hi));
        Self { hi, lo }
    }

    fn times_f(self, value: f64) -> Self {
        let (product, error) = two_prod(self.hi, value);
        let (hi, lo) = quick_two_sum(product, error + self.lo * value);
        Self { hi, lo }
    }

    fn over(self, other: Self) -> Self {
        let first = self.hi / other.hi;
        let remainder = self.minus(other.times_f(first));
        let second = remainder.hi / other.hi;
        let remainder = remainder.minus(other.times_f(second));
        let third = remainder.hi / other.hi;
        let (hi, lo) = quick_two_sum(first, second);
        Self { hi, lo }.plus_f(third)
    }

    fn scaled(self, power: f64) -> Self {
        Self {
            hi: self.hi * power,
            lo: self.lo * power,
        }
    }
}

/// Three float64s, about 159 bits.
#[derive(Clone, Copy, Debug)]
struct Triple {
    hi: f64,
    mid: f64,
    lo: f64,
}

/// `a + b + c` as a triple, the three of decreasing size: exact, each part
/// left within about half an ulp of the one before.
#[inline]
fn renormalized(a: f64, b: f64, c: f64) -> Triple {
    let (b, c) = two_sum(b, c);
    let (hi, b) = two_sum(a, b);
    let (mid, lo) = two_sum(b, c);
    Triple { hi, mid, lo }
}

impl Wide for Triple {
    const ZERO: Self = Self {
        hi: 0.0,
        mid: 0.0,
        lo: 0.0,
    };
    const ONE: Self = Self {
        hi: 1.0,
        mid: 0.0,
        lo: 0.0,
    };
    const EXPANSION_MIN: f64 = 40.0;
    const LOG_TERMS: (usize, usize) = (21, 32);
    const SINE_TERMS: (usize, usize) = (14, 20);
    const EXPANSION_TERMS: (usize, usize) = (10, 20);

    fn from_table(value: [f64; 3]) -> Self {
        Self {
            hi: value[0],
            mid: value[1],
            lo: value[2],
        }
    }

    fn of(value: f64) -> Self {
        Self {
            hi: value,
            ..Self::ZERO
        }
    }

    fn head(self) -> f64 {
        self.hi
    }

    fn value(self) -> f64 {
        self.hi + (self.mid + self.lo)
    }

    fn negated(self) -> Self {
        Self {
            hi: -self.hi,
            mid: -self.mid,
            lo: -self.lo,
        }
    }

    #[inline(never)]
    fn plus(self, other: Self) -> Self {
        let (sum, error) = two_sum(self.hi, other.hi);
        let (middle, middle_error) = two_sum(self.mid, other.mid);
        let (middle, carried) = two_sum(middle, error);
        // Each part of the third under 2^-100 of the whole, so that its
        // roundings are under 2^-150 of it.
        let low = ((self.lo + other.lo) + middle_error) + carried;
        renormalized(sum, middle, low)
    }

    #[inline(never)]
    fn plus_f(self, value: f64) -> Self {
        let (sum, error) = two_sum(self.hi, value);
        let (middle, carried) = two_sum(self.mid, error);
        renormalized(sum, middle, self.lo + carried)
    }

    #[inline(never)]
    fn times(self, other: Self) -> Self {
        let (product, error) = two_prod(self.hi, other.hi);
        let (first, first_error) = two_prod(self.hi, other.mid);
        let (second, second_error) = two_prod(self.mid, other.hi);
        let (middle, middle_error) = two_sum(first, second);
        let (middle, carried) = two_sum(middle, error);
        // The products past these are under 2^-150 of the whole.
        let low = (first_error + second_error)
            + (self.hi * other.lo + self.mid * other.mid + self.lo * other.hi)
            + (middle_error + carried);
        renormalized(product, middle, low)
    }

    #[inline(never)]
    fn times_f(self, value: f64) -> Self {
        let (product, error) = two_prod(self.hi, value);
        let (middle, middle_error) = two_prod(self.mid, value);
        let (middle, carried) = two_sum(middle, error);
        renormalized(product, middle, self.lo * value + (middle_error + carried))
    }

    #[inline(never)]
    fn over(self, other: Self) -> Self {
        let first = self.hi / other.hi;
        let remainder = self.minus(other.times_f(first));
        let second = remainder.hi / other.hi;
        let remainder = remainder.minus(other.times_f(second));
        let third = remainder.hi / other.hi;
        let remainder = remainder.minus(other.times_f(third));
        let fourth = remainder.hi / other.hi;
        renormalized(first, second, third + fourth)
    }

    fn scaled(self, power: f64) -> Self {
        Self {
            hi: self.hi * power,
            mid: self.mid * power,
            lo: self.lo * power,
        }
    }
}

/// `ln a` of a positive, normal `a`.
fn ln<W: Wide>(a: W) -> W {
    // `a = 2^e m`, `m` within a factor of `sqrt 2` of one, the scaling exact.
    let head = a.head();
    let mut exponent = ((head.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut scale = 2f64.powi(-exponent);
    if head * scale > std::f64::consts::SQRT_2 {
        exponent += 1;
        scale *= 0.5;
    }
    let m = a.scaled(scale);
    // `ln m = 2 atanh s` with `s = (m - 1)/(m + 1)` below 0.172, and
    // `atanh s = s Σ s^2k / (2k + 1)`.
    let s = m.plus_f(-1.0).over(m.plus_f(1.0));
    let square = s.times(s);
    let (wide, terms) = W::LOG_TERMS;
    let mut tail = 0.0;
    for k in (wide..terms).rev() {
        tail = tail * square.head() + 1.0 / (2 * k + 1) as f64;
    }
    let mut sum = W::of(tail);
    for &coefficient in ODD[..wide].iter().rev() {
        sum = sum.times(square).plus(W::from_table(coefficient));
    }
    W::from_table(LN_2)
        .times_f(f64::from(exponent))
        .plus(s.times(sum).scaled(2.0))
}

/// `sin(pi r)` and `cos(pi r)` for `|r|` at most a half.
fn sin_cos_pi<W: Wide>(r: f64) -> (W, W) {
    let magnitude = r.abs();
    // Past a quarter, the cosine and the sine of the complement, which is
    // exact.
    let (reduced, swapped) = if magnitude > 0.25 {
        (0.5 - magnitude, true)
    } else {
        (magnitude, false)
    };
    let t = W::from_table(PI).times_f(reduced);
    let square = t.times(t);
    // `sin t = t Σ (-1)^k t^2k / (2k + 1)!` and `cos t = Σ (-1)^k t^2k /
    // (2k)!` for `t` up to `pi/4`.
    let (wide, terms) = W::SINE_TERMS;
    let (mut sine_tail, mut cosine_tail) = (0.0, 0.0);
    for k in (wide..terms).rev() {
        sine_tail = sine_tail * square.head() + SINE[k][0];
        cosine_tail = cosine_tail * square.head() + COSINE[k][0];
    }
    let (mut sine, mut cosine) = (W::of(sine_tail), W::of(cosine_tail));
    for k in (0..wide).rev() {
        sine = sine.times(square).plus(W::from_table(SINE[k]));
        cosine = cosine.times(square).plus(W::from_table(COSINE[k]));
    }
    let sine = t.times(sine);
    let (sine, cosine) = if swapped {
        (cosine, sine)
    } else {
        (sine, cosine)
    };
    (if r < 0.0 { sine.negated() } else { sine }, cosine)
}

/// `Σ c_k u^k` for `k` from one, `c_k` from `coefficients`.
fn expansion<W: Wide>(coefficients: &[[f64; 3]; 20], u: W) -> W {
    let (wide, terms) = W::EXPANSION_TERMS;
    let mut tail = 0.0;
    for coefficient in coefficients[wide..terms].iter().rev() {
        tail = tail * u.head() + coefficient[0];
    }
    let mut sum = W::of(tail);
    for &coefficient in coefficients[..wide].iter().rev() {
        sum = sum.times(u).plus(W::from_table(coefficient));
    }
    sum.times(u)
}

/// `digamma(y)` for a positive `y`, and the magnitude of the terms it is the
/// difference of: `digamma(y + n) - Σ 1/(y + i)`, with `y + n` where the
/// asymptotic expansion holds.
fn digamma_positive<W: Wide>(y: f64) -> (W, f64) {
    let mut w = W::of(y);
    // The steps' sum as one fraction, without a division at each.
    let (mut numerator, mut denominator) = (W::ZERO, W::ONE);
    while w.head() < W::EXPANSION_MIN {
        numerator = numerator.times(w).plus(denominator);
        denominator = denominator.times(w);
        w = w.plus_f(1.0);
    }
    let steps = numerator.over(denominator);
    // `digamma(w) = ln w - 1/(2w) - Σ B_2k / (2k w^2k)`.
    let inverse = w.reciprocal();
    let log = ln(w);
    let value = log
        .minus(inverse.scaled(0.5))
        .minus(expansion(&DIGAMMA, inverse.times(inverse)))
        .minus(steps);
    (value, log.magnitude() + steps.magnitude())
}

/// `lgamma(y)` for a positive `y`, and the magnitude of the terms it is the
/// difference of: `lgamma(y + n) - ln Π (y + i)`, with `y + n` where
/// Stirling's series holds.
fn lgamma_positive<W: Wide>(y: f64) -> (W, f64) {
    let mut w = W::of(y);
    let mut product = W::ONE;
    while w.head() < W::EXPANSION_MIN {
        product = product.times(w);
        w = w.plus_f(1.0);
    }
    // `lgamma(w) = (w - 1/2) ln w - w + ln(2 pi)/2 + Σ B_2k w^(1 - 2k) /
    // (2k (2k - 1))`.
    let inverse = w.reciprocal();
    let leading = w.plus_f(-0.5).times(ln(w));
    let shift = ln(product);
    let value = leading
        .minus(w)
        .plus(W::from_table(HALF_LN_TWO_PI))
        .plus(expansion(&LGAMMA, inverse.times(inverse)).times(w))
        .minus(shift);
    (
        value,
        leading.magnitude() + w.magnitude() + shift.magnitude(),
    )
}

/// `digamma(x) = digamma(-x) - 1/x - pi cot(pi x)` for a negative `x` that
/// is not an integer, and the magnitude of its terms.
fn digamma_with<W: Wide>(x: f64) -> (W, f64) {
    let (sine, cosine) = sin_cos_pi::<W>(x - x.round());
    let cotangent = W::from_table(PI).times(cosine).over(sine);
    let (psi, magnitude) = digamma_positive::<W>(-x);
    let reciprocal = W::of(x).reciprocal();
    (
        psi.minus(reciprocal).minus(cotangent),
        magnitude + reciprocal.magnitude() + cotangent.magnitude(),
    )
}

/// `lgamma(x) = ln pi - ln |x sin(pi x)| - lgamma(-x)` for `x` below minus
/// one and not an integer, and the magnitude of its terms.
fn lgamma_with<W: Wide>(x: f64) -> (W, f64) {
    let (sine, _) = sin_cos_pi::<W>(x - x.round());
    let product = sine.times_f(x);
    let log = ln(if product.head() < 0.0 {
        product.negated()
    } else {
        product
    });
    let (gamma, magnitude) = lgamma_positive::<W>(-x);
    let ln_pi = W::from_table(LN_PI);
    (
        ln_pi.minus(log).minus(gamma),
        magnitude + ln_pi.magnitude() + log.magnitude(),
    )
}

/// The float64 nearest the answer `pair` holds, where its bound places it;
/// where it does not, `triple`'s.
fn settled(pair: (Pair, f64), triple: fn(f64) -> f64, x: f64) -> f64 {
    let (value, magnitude) = pair;
    <f64 as Output>::settle(value.hi, value.lo, magnitude * PAIR_ERROR).unwrap_or_else(|| triple(x))
}

/// The triples' answers, which only the floats next to a zero need: out of
/// line, their arithmetic too, since what they cost is not what matters.
#[cold]
#[inline(never)]
fn digamma_triple(x: f64) -> f64 {
    digamma_with::<Triple>(x).0.value()
}

#[cold]
#[inline(never)]
fn lgamma_triple(x: f64) -> f64 {
    lgamma_with::<Triple>(x).0.value()
}

/// `digamma(x)` for a negative `x` that is not an integer, rounded once.
pub(super) fn digamma_reflected(x: f64) -> f64 {
    settled(digamma_with::<Pair>(x), digamma_triple, x)
}

/// `lgamma(x)` for `x` below minus one and not an integer, rounded once.
pub(super) fn lgamma_reflected(x: f64) -> f64 {
    settled(lgamma_with::<Pair>(x), lgamma_triple, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settles(pair: (Pair, f64)) -> bool {
        let (value, magnitude) = pair;
        <f64 as Output>::settle(value.hi, value.lo, magnitude * PAIR_ERROR).is_some()
    }

    /// At the floats nearest the zero of `lgamma` at -2.457 and that of
    /// `digamma` at -0.504 the answers are 5.6e-17 and 7.3e-17 against terms
    /// of size one: the pair's bound does not place them, and the triple's
    /// answer is the float nearest the exact value.
    #[test]
    fn next_to_a_zero_the_triple_answers() {
        let x = -2.457_024_738_220_800_6;
        assert!(!settles(lgamma_with::<Pair>(x)));
        assert_eq!(lgamma_reflected(x), 5.619_192_358_950_097e-17);
        let x = -0.504_083_008_264_455_4;
        assert!(!settles(digamma_with::<Pair>(x)));
        assert_eq!(digamma_reflected(x), 7.289_763_902_976_895e-17);
    }

    /// Where the pair settles, the triple rounds to the same float.
    #[test]
    fn the_pair_and_the_triple_agree() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut settled = 0;
        for _ in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = -1.0 - 40.0 * ((state >> 11) as f64 / (1u64 << 53) as f64);
            if x == x.floor() {
                continue;
            }
            for (pair, triple) in [
                (digamma_with::<Pair>(x), digamma_with::<Triple>(x)),
                (lgamma_with::<Pair>(x), lgamma_with::<Triple>(x)),
            ] {
                let (value, magnitude) = pair;
                if let Some(answer) =
                    <f64 as Output>::settle(value.hi, value.lo, magnitude * PAIR_ERROR)
                {
                    settled += 1;
                    assert_eq!(answer, triple.0.value(), "{x}");
                }
            }
        }
        assert!(settled > 3900, "{settled}");
    }
}
