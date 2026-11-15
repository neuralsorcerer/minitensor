// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Determinants of order two to four, and Cramer's quotients of them,
//! rounded once.
//!
//! A determinant of order two is the sum of two products of its entries, one
//! of order three the sum of six and one of order four of twenty-four, and by
//! Cramer's rule each entry of the solution of a system of that order -- and
//! so of an inverse, the solution against the identity -- is the quotient of
//! two of them. Taken by elimination, each pivot, product and update rounded,
//! a determinant of whole numbers came out -0.9999999999999998 where it is -1.
//!
//! Here each is the float nearest its exact value. The products are split
//! exactly and added with a bound on what the additions rounded, and a
//! quotient takes its correction from the exact remainder of its leading
//! division; where the bound places the value it is taken from the pair, and
//! where it does not, or an entry lies too far out for the products to split
//! exactly, the sums are taken in whole numbers.
//!
//! The cofactors of a row are worked out side by side, four lanes of the
//! same arithmetic, and so are the entries of a column of a solution.

use super::{Output, Total, product_error, two_sum};

/// A unit in the last place of float64, relative: what one rounding to
/// nearest can move a value by.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// What the additions of [`combine`] and [`combine_lanes`] round, in units
/// of the magnitude of what they add: ten roundings at most, each of at most
/// a unit, with room for the roundings of the magnitudes themselves.
const ROUNDINGS: f64 = 14.0 * UNIT;

/// A value within `bound` of the exact one, as a two-sum's pair.
#[derive(Clone, Copy, Debug)]
struct Estimate {
    value: f64,
    rest: f64,
    bound: f64,
}

impl Estimate {
    const ZERO: Self = Self {
        value: 0.0,
        rest: 0.0,
        bound: 0.0,
    };
}

/// Four values side by side.
type Four = [f64; 4];

/// Four estimates side by side, one in each lane.
#[derive(Clone, Copy, Debug)]
struct Lanes {
    value: Four,
    rest: Four,
    bound: Four,
}

impl Lanes {
    const ZERO: Self = Self {
        value: [0.0; 4],
        rest: [0.0; 4],
        bound: [0.0; 4],
    };

    /// Exact values.
    fn exact(value: Four) -> Self {
        Self {
            value,
            ..Self::ZERO
        }
    }

    /// The estimate in lane `l`.
    fn lane(&self, l: usize) -> Estimate {
        Estimate {
            value: self.value[l],
            rest: self.rest[l],
            bound: self.bound[l],
        }
    }

    /// The estimates `picks` of `from`, one to a lane.
    fn picked(from: &[Estimate; 6], picks: [usize; 4]) -> Self {
        let mut out = Self::ZERO;
        for l in 0..4 {
            let estimate = from[picks[l]];
            out.value[l] = estimate.value;
            out.rest[l] = estimate.rest;
            out.bound[l] = estimate.bound;
        }
        out
    }
}

/// A bound grown a little past itself, so that the roundings of its own
/// arithmetic cannot leave it short.
#[inline(always)]
fn generous(bound: f64) -> f64 {
    bound * (1.0 + 2f64.powi(-40))
}

/// Whether every entry lies far enough inside the range that the products the
/// estimates split -- of up to four entries -- their errors and the products
/// of those stay ordinary float64s: each entry zero, or with an exponent
/// within 180 of one's, a magnitude from `2^-180` to below `2^181`.
fn comfortable(entries: &[f64]) -> bool {
    // Without a branch, so that it runs as fast as the entries load.
    entries.iter().fold(true, |comfortable, &value| {
        let exponent = (value.to_bits() >> 52) & 0x7ff;
        comfortable & ((value == 0.0) | (exponent.wrapping_sub(1023 - 180) <= 360))
    })
}

/// Whether a fused multiply-add is in the baseline of the target, as it is on
/// aarch64; on x86_64 it is detected where the work is dispatched.
const FUSED: bool = cfg!(target_arch = "aarch64");

/// A function taking `FMA` true where the CPU fuses a multiply and an add,
/// compiled for that feature, and false where it does not: the product
/// errors the estimates split are one instruction with it and some fifteen
/// without. What the function calls with it is inlined, so that it is
/// compiled with the feature too; a closure handed on to a function that is
/// not would be compiled without it.
macro_rules! fused {
    ($(#[$doc:meta])* fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $out:ty = $with:ident) => {
        $(#[$doc])*
        fn $name($($arg: $ty),*) -> $out {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("fma") {
                #[target_feature(enable = "fma")]
                fn fused($($arg: $ty),*) -> $out {
                    $with::<true>($($arg),*)
                }
                // SAFETY: the feature was detected on this CPU just above.
                return unsafe { fused($($arg),*) };
            }
            $with::<FUSED>($($arg),*)
        }
    };
}

/// `x · y` and its exact error.
#[inline(always)]
fn split<const FMA: bool>(x: f64, y: f64) -> (f64, f64) {
    let product = x * y;
    let error = if FMA {
        x.mul_add(y, -product)
    } else {
        product_error(x, y, product)
    };
    (product, error)
}

/// `x · y - z · w`.
#[inline(always)]
fn minor<const FMA: bool>(x: f64, y: f64, z: f64, w: f64) -> Estimate {
    let (left, left_error) = split::<FMA>(x, y);
    let (right, right_error) = split::<FMA>(z, w);
    let (head, carry) = two_sum(left, -right);
    // Two roundings: of the errors' difference, and of its sum with what
    // the two-sum left.
    let errors = left_error - right_error;
    let tail = carry + errors;
    let (value, rest) = two_sum(head, tail);
    Estimate {
        value,
        rest,
        bound: generous((errors.abs() + tail.abs()) * UNIT),
    }
}

/// [`minor`] lane by lane.
#[inline(always)]
fn minor_lanes<const FMA: bool>(x: Four, y: Four, z: Four, w: Four) -> Lanes {
    let mut out = Lanes::ZERO;
    for l in 0..4 {
        let estimate = minor::<FMA>(x[l], y[l], z[l], w[l]);
        out.value[l] = estimate.value;
        out.rest[l] = estimate.rest;
        out.bound[l] = estimate.bound;
    }
    out
}

/// One term `x · e` of a sum: its head and what it adds to the sum's small
/// part and to its bound -- what the estimate's own bound and the rounding
/// of `x` times its low part leave, and what adding the small parts rounds.
#[inline(always)]
fn term<const FMA: bool>(x: f64, e: Estimate) -> (f64, f64, f64) {
    let (product, product_error) = split::<FMA>(x, e.value);
    let carried = x * e.rest;
    let bound =
        x.abs() * e.bound + carried.abs() * (UNIT + ROUNDINGS) + product_error.abs() * ROUNDINGS;
    (product, product_error + carried, bound)
}

/// `Σ x · e` over entries `xs` and estimates `es`: a zero entry adds
/// nothing.
#[inline(always)]
fn combine<const FMA: bool, const T: usize>(xs: [f64; T], es: [Estimate; T]) -> Estimate {
    let (mut total, mut small, mut bound) = (0.0f64, 0.0f64, 0.0f64);
    for t in 0..T {
        if xs[t] == 0.0 {
            continue;
        }
        let (head, low, more) = term::<FMA>(xs[t], es[t]);
        let (sum, error) = two_sum(total, head);
        total = sum;
        small += low + error;
        bound += more + error.abs() * ROUNDINGS;
    }
    let (value, rest) = two_sum(total, small);
    Estimate {
        value,
        rest,
        bound: generous(bound),
    }
}

/// [`combine`] lane by lane: `Σ x · e` over the entries `xs` and the
/// estimates `es` of each lane, a term whose entries are all zero passed by.
#[inline(always)]
fn combine_lanes<const FMA: bool, const T: usize>(xs: [Four; T], es: [Lanes; T]) -> Lanes {
    let (mut total, mut small, mut bound) = ([0.0f64; 4], [0.0f64; 4], [0.0f64; 4]);
    for t in 0..T {
        if xs[t] == [0.0; 4] {
            continue;
        }
        for l in 0..4 {
            let (head, low, more) = term::<FMA>(xs[t][l], es[t].lane(l));
            let (sum, error) = two_sum(total[l], head);
            total[l] = sum;
            small[l] += low + error;
            bound[l] += more + error.abs() * ROUNDINGS;
        }
    }
    let mut out = Lanes::ZERO;
    for l in 0..4 {
        let (value, rest) = two_sum(total[l], small[l]);
        out.value[l] = value;
        out.rest[l] = rest;
        out.bound[l] = generous(bound[l]);
    }
    out
}

/// The cofactor of entry `(r, i)` of the 3x3 matrix `m`, rows in order, as
/// `x · y - z · w`: the minor without row `r` and column `i`, its sign taken
/// by swapping the two products, which negates exactly.
#[inline(always)]
fn cofactor3_entries(m: &[f64], r: usize, i: usize) -> (f64, f64, f64, f64) {
    let (top, bottom) = ([1, 0, 0][r], [2, 2, 1][r]);
    let (left, right) = ([1, 0, 0][i], [2, 2, 1][i]);
    let at = |row: usize, column: usize| m[row * 3 + column];
    let (a, b) = (at(top, left), at(bottom, right));
    let (c, d) = (at(top, right), at(bottom, left));
    if (r + i) % 2 == 1 {
        (c, d, a, b)
    } else {
        (a, b, c, d)
    }
}

/// The pairs of columns of a 4x4 matrix, in order.
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// The 2x2 minors of rows `top` and `top + 1` of the 4x4 matrix `m`, at
/// each pair of columns.
#[inline(always)]
fn row_pair_minors<const FMA: bool>(m: &[f64], top: usize) -> [Estimate; 6] {
    let at = |row: usize, column: usize| m[row * 4 + column];
    let mut minors = [Estimate::ZERO; 6];
    for (slot, &(p, q)) in minors.iter_mut().zip(&PAIRS) {
        *slot = minor::<FMA>(at(top, p), at(top + 1, q), at(top, q), at(top + 1, p));
    }
    minors
}

/// The three 2x2 minors each cofactor of a row of a 4x4 matrix takes, a lane
/// for each column, from the minors of a pair of rows at [`PAIRS`]: of the
/// columns other than `j`, `k0 < k1 < k2`, the minors at `(k1, k2)`,
/// `(k0, k2)` and `(k0, k1)`.
#[inline(always)]
fn complements(minors: &[Estimate; 6]) -> [Lanes; 3] {
    [
        Lanes::picked(minors, [5, 5, 4, 3]),
        Lanes::picked(minors, [4, 2, 2, 1]),
        Lanes::picked(minors, [3, 1, 0, 0]),
    ]
}

/// Row `i` of the cofactors of the 4x4 matrix `m`, a lane for each column:
/// the 3x3 minor without row `i` and column `j`, expanded along the row of
/// it outside the pair of rows whose minors `complements` holds -- rows two
/// and three for `i` zero and one, rows zero and one for `i` two and three --
/// and signed as the cofactor is, in the entries, which negates exactly.
#[inline(always)]
fn cofactor_row4<const FMA: bool>(m: &[f64], i: usize, complements: &[Lanes; 3]) -> Lanes {
    let row = [1, 0, 3, 2][i];
    // Of the columns other than `j`, the first, the second and the third.
    const COLUMNS: [[usize; 4]; 3] = [[1, 0, 0, 0], [2, 2, 1, 1], [3, 3, 3, 2]];
    let mut xs = [[0.0f64; 4]; 3];
    for t in 0..3 {
        for j in 0..4 {
            let entry = m[row * 4 + COLUMNS[t][j]];
            xs[t][j] = if (i + j + t) % 2 == 1 { -entry } else { entry };
        }
    }
    combine_lanes::<FMA, 3>(xs, *complements)
}

/// The cofactors of the `n`x`n` matrix `m`, rows in order, a row of the
/// cofactor matrix to each `Lanes` and a column to each lane, its
/// determinant, and the float64 nearest the determinant's reciprocal.
struct Adjugate {
    rows: [Lanes; 4],
    determinant: Estimate,
    reciprocal: f64,
}

/// The cofactors of the `n`x`n` matrix `m` in its first `count` rows at
/// least, the rest zero, and its determinant along the first row.
#[inline(always)]
fn adjugate_with<const FMA: bool>(m: &[f64], n: usize, count: usize) -> Adjugate {
    let mut rows = [Lanes::ZERO; 4];
    match n {
        2 => {
            rows[0] = Lanes::exact([m[3], -m[2], 0.0, 0.0]);
            rows[1] = Lanes::exact([-m[1], m[0], 0.0, 0.0]);
        }
        3 => {
            for i in 0..count {
                let mut entries = [[0.0f64; 4]; 4];
                for j in 0..3 {
                    let (x, y, z, w) = cofactor3_entries(m, i, j);
                    entries[0][j] = x;
                    entries[1][j] = y;
                    entries[2][j] = z;
                    entries[3][j] = w;
                }
                let [x, y, z, w] = entries;
                rows[i] = minor_lanes::<FMA>(x, y, z, w);
            }
        }
        _ => {
            // The first two rows take the minors of the last two rows of
            // `m`, and the last two those of the first two.
            let mut sets = [[Lanes::ZERO; 3]; 2];
            for (set, top) in sets.iter_mut().zip([2, 0]).take(count.div_ceil(2)) {
                *set = complements(&row_pair_minors::<FMA>(m, top));
            }
            for i in 0..count {
                rows[i] = cofactor_row4::<FMA>(m, i, &sets[i / 2]);
            }
        }
    }
    // Along the first row.
    let mut first = [0.0f64; 4];
    first[..n].copy_from_slice(&m[..n]);
    let cofactors = [0, 1, 2, 3].map(|j| rows[0].lane(j));
    let determinant = combine::<FMA, 4>(first, cofactors);
    Adjugate {
        rows,
        determinant,
        reciprocal: 1.0 / determinant.value,
    }
}

fused! {
    /// [`adjugate_with`], `n` two to four and the entries comfortable.
    fn adjugate(m: &[f64], n: usize, count: usize) -> Adjugate = adjugate_with
}

/// The determinant of the `n`x`n` matrix `m`, rows in order, `n` two or
/// three, its entries comfortable.
#[inline(always)]
fn small_estimate_with<const FMA: bool>(m: &[f64], n: usize) -> Estimate {
    if n == 2 {
        return minor::<FMA>(m[0], m[3], m[1], m[2]);
    }
    let cofactor = |i: usize| {
        let (x, y, z, w) = cofactor3_entries(m, 0, i);
        minor::<FMA>(x, y, z, w)
    };
    combine::<FMA, 3>([m[0], m[1], m[2]], [cofactor(0), cofactor(1), cofactor(2)])
}

fused! {
    /// [`small_estimate_with`].
    fn small_estimate(m: &[f64], n: usize) -> Estimate = small_estimate_with
}

/// The determinant of the `n`x`n` matrix `m`, rows in order, `n` two to four,
/// its entries comfortable: of order four, along the first row of the
/// cofactors.
fn estimate(m: &[f64], n: usize) -> Estimate {
    if n == 4 {
        adjugate(m, n, 1).determinant
    } else {
        small_estimate(m, n)
    }
}

/// `N / D` lane by lane, from `D`'s estimate and the float64 nearest its
/// reciprocal; each lane with whether it holds: not where the quotient
/// leaves the ordinary range or `D`'s bound reaches near its value.
#[inline(always)]
fn quotient_lanes<const FMA: bool>(
    numerator: Lanes,
    divisor: Estimate,
    reciprocal: f64,
) -> ([Estimate; 4], [bool; 4]) {
    let scale = divisor.value.abs();
    // Written so that a NaN fails it too.
    let placed = divisor.bound < scale * 2f64.powi(-50);
    // At least one over the exact divisor's magnitude: that is within
    // `2^-49` of `scale` in proportion -- its bound under `2^-50` of it and
    // its low part under a rounding's -- and `reciprocal` within a rounding
    // of one over `scale`.
    let inverse = reciprocal.abs() * (1.0 + 2f64.powi(-47));
    let mut estimates = [Estimate::ZERO; 4];
    let mut holds = [false; 4];
    for l in 0..4 {
        let (value, rest, bound) = (numerator.value[l], numerator.rest[l], numerator.bound[l]);
        // A leading quotient within two roundings of `N`'s leading part over
        // `D`'s, and its remainder: `N - product` is exact, the two being
        // that near, and taking the product's error from it rounds once.
        let leading = value * reciprocal;
        let (product, product_error) = split::<FMA>(leading, divisor.value);
        let remainder = (value - product) - product_error;
        // What the pairs' low parts add to it, in two roundings and a third
        // for the product.
        let carried = leading * divisor.rest;
        let low = rest - carried;
        let residual = remainder + low;
        // Within three roundings of `residual` over `D`'s leading part.
        let correction = residual * reciprocal;
        // The exact divisor is within its bound and its low part of the
        // leading one, so that dividing by the leading one moves the
        // correction by those over it; the estimates' bounds and the
        // roundings move the residual. A rounding below float64's normal
        // range moves a value by up to half its spacing there, not by a
        // part of the value: the last term, which the division scales as it
        // scales the rest, is for the four that can land there.
        let moved = bound
            + leading.abs() * divisor.bound
            + (remainder.abs() + carried.abs() + low.abs() + residual.abs()) * UNIT
            + residual.abs() * (divisor.rest.abs() + divisor.bound) * inverse
            + f64::from_bits(4);
        let (sum, error) = two_sum(leading, correction);
        let quotient = Estimate {
            value: sum,
            rest: error,
            bound: generous(moved * inverse + correction.abs() * (3.0 * UNIT)) + f64::from_bits(2),
        };
        // An exact zero over anything not zero is an exact zero.
        let zero = value == 0.0 && bound == 0.0;
        estimates[l] = if zero { Estimate::ZERO } else { quotient };
        holds[l] = placed & (zero | (2f64.powi(-900)..=2f64.powi(900)).contains(&leading.abs()));
    }
    (estimates, holds)
}

/// Entries `0..n` of the solution of `m x = b`, `b`'s column `column` of
/// `columns`, as estimates: of entry `i`, the determinant of `m` with its
/// column `i` taken from `b` -- `b` against the cofactors of that column --
/// over the determinant of `m`. None for each past `n`, or where the
/// estimate does not hold.
#[inline(always)]
fn column_quotients_with<const FMA: bool>(
    adjugate: &Adjugate,
    n: usize,
    b: &[f64],
    columns: usize,
    column: usize,
) -> [Option<Estimate>; 4] {
    let mut xs = [[0.0f64; 4]; 4];
    for r in 0..n {
        xs[r] = [b[r * columns + column]; 4];
    }
    let numerators = combine_lanes::<FMA, 4>(xs, adjugate.rows);
    let (estimates, holds) =
        quotient_lanes::<FMA>(numerators, adjugate.determinant, adjugate.reciprocal);
    let mut out = [None; 4];
    for i in 0..n {
        out[i] = holds[i].then_some(estimates[i]);
    }
    out
}

fused! {
    /// [`column_quotients_with`].
    fn column_quotients(
        adjugate: &Adjugate,
        n: usize,
        b: &[f64],
        columns: usize,
        column: usize,
    ) -> [Option<Estimate>; 4] = column_quotients_with
}

/// `scale` times each cofactor of `adjugate` of order `n`, rows in order.
#[inline(always)]
fn scaled_with<const FMA: bool>(adjugate: &Adjugate, n: usize, scale: f64) -> [Estimate; 16] {
    let mut out = [Estimate::ZERO; 16];
    for r in 0..n {
        let row = adjugate.rows[r];
        let mut scaled = Lanes::ZERO;
        for l in 0..4 {
            let (product, product_error) = split::<FMA>(scale, row.value[l]);
            let carried = scale * row.rest[l];
            let tail = product_error + carried;
            let (value, rest) = two_sum(product, tail);
            scaled.value[l] = value;
            scaled.rest[l] = rest;
            scaled.bound[l] =
                generous(scale.abs() * row.bound[l] + (carried.abs() + tail.abs()) * UNIT);
        }
        for j in 0..n {
            out[r * n + j] = scaled.lane(j);
        }
    }
    out
}

fused! {
    /// [`scaled_with`].
    fn scaled(adjugate: &Adjugate, n: usize, scale: f64) -> [Estimate; 16] = scaled_with
}

/// The exact determinant of the `n`x`n` matrix `m` as a sum of products, one
/// for each permutation, four factors each with ones past the order and the
/// permutation's sign on the first.
fn terms(m: &[f64], n: usize) -> Vec<[f64; 4]> {
    fn permute(order: &mut [usize], k: usize, m: &[f64], n: usize, out: &mut Vec<[f64; 4]>) {
        if k == n {
            let mut factors = [1.0f64; 4];
            for (row, factor) in factors.iter_mut().take(n).enumerate() {
                *factor = m[row * n + order[row]];
            }
            let inversions = (0..n)
                .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                .filter(|&(a, b)| order[a] > order[b])
                .count();
            if inversions % 2 == 1 {
                factors[0] = -factors[0];
            }
            out.push(factors);
            return;
        }
        for i in k..n {
            order.swap(k, i);
            permute(order, k + 1, m, n, out);
            order.swap(k, i);
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    permute(&mut order, 0, m, n, &mut out);
    out
}

/// The float nearest the exact sum of `terms`; an exact zero is positive.
#[cold]
#[inline(never)]
fn exact_value<T: Output>(terms: &[[f64; 4]]) -> T {
    let total = Total::of_quads(terms);
    if total.positive == total.negative {
        return T::narrow(0.0);
    }
    total.rounded()
}

/// The float nearest the exact quotient of the sums of `numerator` and
/// `divisor`, the divisor not zero; an exact zero is positive.
#[cold]
#[inline(never)]
fn exact_quotient<T: Output>(numerator: &[[f64; 4]], divisor: &[[f64; 4]]) -> T {
    let total = Total::of_quads(numerator);
    if total.positive == total.negative {
        return T::narrow(0.0);
    }
    total.quotient(Total::of_quads(divisor))
}

/// Whether the exact determinant of `m` is zero, its entries finite.
fn singular(m: &[f64], n: usize) -> bool {
    let total = Total::of_quads(&terms(m, n));
    total.positive == total.negative
}

/// The value an estimate settles to, where it does: an estimate of zero
/// settles only with no bound, as an exact zero, which is positive.
fn settled<T: Output>(estimate: Option<Estimate>) -> Option<T> {
    let estimate = estimate?;
    let value = T::settle(estimate.value, estimate.rest, estimate.bound)?;
    Some(if estimate.value == 0.0 {
        T::narrow(0.0)
    } else {
        value
    })
}

/// The float nearest the exact determinant of the `n`x`n` matrix `m`, rows in
/// order, `n` two to four and every entry finite; a zero determinant is
/// positive zero.
pub(crate) fn small_determinant<T: Output>(m: &[f64], n: usize) -> T {
    settled(comfortable(m).then(|| estimate(m, n))).unwrap_or_else(|| exact_value(&terms(m, n)))
}

/// The solution of `a x = b` into `out`, each entry the float nearest its
/// exact value: `a` is `n`x`n` and `b` and `out` are `n`x`columns`, rows in
/// order, `n` two to four and every entry finite. False, with `out` as it
/// was, where `a` is singular.
pub(crate) fn small_solve<T: Output>(
    a: &[f64],
    n: usize,
    b: &[f64],
    columns: usize,
    out: &mut [T],
) -> bool {
    // Cramer's rule through the adjugate: entry `i` is the determinant of
    // `a` with its column `i` taken from `b`, which is `b` against the
    // cofactors of that column, over the determinant of `a`.
    let quick = (comfortable(a) && comfortable(b)).then(|| adjugate(a, n, n));
    let nonzero = quick.as_ref().is_some_and(|quick| {
        let d = quick.determinant;
        d.value.abs() > d.rest.abs() + d.bound
    });
    if !nonzero && singular(a, n) {
        return false;
    }
    for column in 0..columns {
        let estimates = quick.as_ref().map_or([None; 4], |quick| {
            column_quotients(quick, n, b, columns, column)
        });
        for (i, estimate) in estimates.into_iter().enumerate().take(n) {
            out[i * columns + column] = settled(estimate).unwrap_or_else(|| {
                exact_quotient(&replaced_terms(a, n, b, columns, column, i), &terms(a, n))
            });
        }
    }
    true
}

/// The terms of the determinant of `a` with its column `i` taken from
/// column `column` of `b`.
#[inline(never)]
fn replaced_terms(
    a: &[f64],
    n: usize,
    b: &[f64],
    columns: usize,
    column: usize,
    i: usize,
) -> Vec<[f64; 4]> {
    let mut replaced = a.to_vec();
    for row in 0..n {
        replaced[row * n + i] = b[row * columns + column];
    }
    terms(&replaced, n)
}

/// The sign of the exact determinant of `m`, `n` two to four and every entry
/// finite, and the logarithm of its magnitude: `(0, -inf)` for a singular
/// matrix. None where the determinant lies outside float64's range even with
/// the entries scaled by a power of two.
pub(crate) fn small_log_determinant(m: &[f64], n: usize) -> Option<(f64, f64)> {
    let quick = comfortable(m).then(|| estimate(m, n));
    let determinant: f64 = settled(quick).unwrap_or_else(|| exact_value(&terms(m, n)));
    if determinant.is_normal() {
        let sign = determinant.signum();
        let magnitude = determinant.abs();
        let log = if (0.5..=2.0).contains(&magnitude) {
            // Near one the logarithm is small, and `ln` of the rounded value
            // would keep only its rounding: `log1p` of the magnitude less
            // one, rounded once.
            let less_one = quick.and_then(|estimate| {
                let (head, carry) = two_sum(sign * estimate.value, -1.0);
                let tail = carry + sign * estimate.rest;
                let (value, rest) = two_sum(head, tail);
                f64::settle(value, rest, generous(estimate.bound + tail.abs() * UNIT))
            });
            less_one
                .unwrap_or_else(|| {
                    let mut shifted = terms(m, n);
                    for term in &mut shifted {
                        term[0] *= sign;
                    }
                    shifted.push([-1.0, 1.0, 1.0, 1.0]);
                    exact_value(&shifted)
                })
                .ln_1p()
        } else {
            magnitude.ln()
        };
        return Some((sign, log));
    }
    if determinant == 0.0 && singular(m, n) {
        return Some((0.0, f64::NEG_INFINITY));
    }
    // Past the range: the entries scaled by a power of two that brings the
    // largest to one, where that is exact, scale the determinant by its
    // `n`-th power.
    let top = m.iter().fold(0.0f64, |top, value| top.max(value.abs()));
    let shift = -(top.log2().floor() as i32);
    let factor = 2f64.powi(shift);
    let back = 2f64.powi(-shift);
    let scaled: Vec<f64> = m.iter().map(|&value| value * factor).collect();
    if !scaled
        .iter()
        .zip(m)
        .all(|(&value, &original)| (value * back == original) & value.is_finite())
    {
        return None;
    }
    let determinant = small_determinant::<f64>(&scaled, n);
    if !determinant.is_normal() {
        return None;
    }
    // `ln 2` in two parts, the leading one short enough that its product
    // with any power here is exact.
    let powers = f64::from(shift) * n as f64;
    let log = (determinant.abs().ln() - powers * 0.693_147_180_369_123_8)
        - powers * 1.908_214_929_270_587_7e-10;
    Some((determinant.signum(), log))
}

/// `scale` times each cofactor of the `n`x`n` matrix `m` into `out`, rows in
/// order, each the float nearest its exact value: the derivative of the
/// determinant, scaled. `n` is two to four and every value finite.
pub(crate) fn small_cofactors<T: Output>(m: &[f64], n: usize, scale: f64, out: &mut [T]) {
    if n == 2 {
        // Each cofactor is one entry, so its product with the scale rounds
        // once as it is: exact in float64 for float32 entries.
        let cofactors = [m[3], -m[2], -m[1], m[0]];
        for (slot, cofactor) in out.iter_mut().zip(cofactors) {
            let product = if scale == 0.0 || cofactor == 0.0 {
                0.0
            } else {
                scale * cofactor
            };
            *slot = T::narrow(product);
        }
        return;
    }
    let estimates = scaled_cofactors(m, n, scale);
    for (index, (slot, estimate)) in out.iter_mut().zip(estimates).enumerate() {
        *slot =
            settled(estimate).unwrap_or_else(|| exact_value(&cofactor_terms(m, n, scale, index)));
    }
}

/// The exact `scale` times cofactor `index` of the `n`x`n` matrix `m`, rows
/// in order, as a sum of products: the minor's, with the scale, signed as
/// the cofactor is, as a factor of each.
fn cofactor_terms(m: &[f64], n: usize, scale: f64, index: usize) -> Vec<[f64; 4]> {
    let (i, j) = (index / n, index % n);
    let minor: Vec<f64> = (0..n)
        .filter(|&row| row != i)
        .flat_map(|row| {
            (0..n)
                .filter(move |&column| column != j)
                .map(move |column| m[row * n + column])
        })
        .collect();
    let signed = if (i + j) % 2 == 1 { -scale } else { scale };
    let mut terms = terms(&minor, n - 1);
    for term in &mut terms {
        term[3] = signed;
    }
    terms
}

/// The estimates of [`small_cofactors`] for orders three and four, the same
/// for every dtype, or None for each where a value lies too far out.
#[inline(never)]
fn scaled_cofactors(m: &[f64], n: usize, scale: f64) -> [Option<Estimate>; 16] {
    if !(comfortable(m) && comfortable(&[scale])) {
        return [None; 16];
    }
    scaled(&adjugate(m, n, n), n, scale).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = ((state >> 3) % 9) as i32 - 4;
            (unit - 0.5) * 2f64.powi(exponent)
        }
    }

    /// Every quick estimate settles to what the whole numbers give.
    #[test]
    fn the_quick_values_are_the_exact_ones() {
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        for trial in 0..600 {
            let n = 2 + trial % 3;
            let mut m: Vec<f64> = (0..n * n).map(|_| next()).collect();
            if trial % 4 == 0 {
                // Whole numbers, whose determinants are whole numbers.
                for value in &mut m {
                    *value = (*value * 16.0).round();
                }
            }
            let want: f64 = exact_value(&terms(&m, n));
            let got: f64 = small_determinant(&m, n);
            assert_eq!(got.to_bits(), want.to_bits(), "{m:?}");
            let narrow: f32 = exact_value(&terms(&m, n));
            assert_eq!(small_determinant::<f32>(&m, n).to_bits(), narrow.to_bits());

            let b: Vec<f64> = (0..n * 2).map(|_| next()).collect();
            let mut out = vec![0.0f64; n * 2];
            if !small_solve(&m, n, &b, 2, &mut out) {
                assert!(singular(&m, n));
                continue;
            }
            for column in 0..2 {
                for i in 0..n {
                    let numerator = replaced_terms(&m, n, &b, 2, column, i);
                    let want: f64 = exact_quotient(&numerator, &terms(&m, n));
                    assert_eq!(out[i * 2 + column].to_bits(), want.to_bits(), "{m:?} {b:?}");
                }
            }

            let scale = next();
            let mut cofactors = vec![0.0f64; n * n];
            small_cofactors(&m, n, scale, &mut cofactors);
            for (index, &got) in cofactors.iter().enumerate() {
                let want: f64 = exact_value(&cofactor_terms(&m, n, scale, index));
                assert_eq!(got.to_bits(), want.to_bits(), "{m:?} {scale} {index}");
            }
        }
    }

    #[test]
    fn whole_numbers_give_whole_numbers() {
        let m = [2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0];
        assert_eq!(small_determinant::<f64>(&m, 3), -1.0);
        let mut inverse = [0.0f64; 9];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(small_solve(&m, 3, &identity, 3, &mut inverse));
        assert_eq!(inverse, [0.0, 0.0, 1.0, -2.0, 1.0, 3.0, 3.0, -1.0, -5.0]);
        // Singular, though elimination leaves a pivot of 6.7e-16.
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(small_determinant::<f64>(&m, 3).to_bits(), 0.0f64.to_bits());
        assert!(!small_solve(&m, 3, &[1.0, 2.0, 3.0], 1, &mut [0.0f64; 3]));
        assert_eq!(small_log_determinant(&m, 3), Some((0.0, f64::NEG_INFINITY)));
        // Of order four: a quarter turn and a translation, whose inverse is
        // the turn's transpose and the translation turned back.
        let m = [
            0.0, -1.0, 0.0, 3.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 1.0,
        ];
        assert_eq!(small_determinant::<f64>(&m, 4), 1.0);
        let mut inverse = [0.0f64; 16];
        let mut identity = [0.0f64; 16];
        for d in 0..4 {
            identity[d * 5] = 1.0;
        }
        assert!(small_solve(&m, 4, &identity, 4, &mut inverse));
        assert_eq!(
            inverse,
            [
                0.0, 1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 0.0, 1.0
            ]
        );
    }

    #[test]
    fn far_entries_are_taken_exactly() {
        // Past the range the estimates split, and a determinant past
        // float64's: the logarithm comes from the entries scaled.
        let m = [1e200, 3e199, 2e199, 1e200];
        assert!(small_determinant::<f64>(&m, 2).is_infinite());
        let (sign, log) = small_log_determinant(&m, 2).expect("in range scaled");
        assert_eq!(sign, 1.0);
        let want = (1.0f64 - 0.06).ln() + 400.0 * 10f64.ln();
        assert!(
            (log - want).abs() <= 4.0 * f64::EPSILON * want,
            "{log} {want}"
        );
    }

    #[test]
    fn cofactors_are_each_rounded_once() {
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        for trial in 0..100 {
            let n = 3 + trial % 2;
            let m: Vec<f64> = (0..n * n).map(|_| next()).collect();
            let scale = next();
            let mut out = [0.0f64; 16];
            small_cofactors(&m, n, scale, &mut out[..n * n]);
            for i in 0..n {
                for j in 0..n {
                    let rows: Vec<usize> = (0..n).filter(|&r| r != i).collect();
                    let columns: Vec<usize> = (0..n).filter(|&c| c != j).collect();
                    let at = |r: usize, c: usize| m[rows[r] * n + columns[c]];
                    let sign = if (i + j) % 2 == 1 { -scale } else { scale };
                    // The minor's products written out, each with the scale.
                    let products: Vec<[f64; 4]> = if n == 3 {
                        vec![
                            [sign, at(0, 0), at(1, 1), 1.0],
                            [-sign, at(0, 1), at(1, 0), 1.0],
                        ]
                    } else {
                        vec![
                            [sign, at(0, 0), at(1, 1), at(2, 2)],
                            [-sign, at(0, 0), at(1, 2), at(2, 1)],
                            [-sign, at(0, 1), at(1, 0), at(2, 2)],
                            [sign, at(0, 1), at(1, 2), at(2, 0)],
                            [sign, at(0, 2), at(1, 0), at(2, 1)],
                            [-sign, at(0, 2), at(1, 1), at(2, 0)],
                        ]
                    };
                    let want: f64 = exact_value(&products);
                    assert_eq!(out[i * n + j].to_bits(), want.to_bits(), "{m:?} {i} {j}");
                }
            }
        }
    }
}
