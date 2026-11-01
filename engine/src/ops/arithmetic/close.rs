// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `isclose`: whether `|a - b| <= atol + rtol · |b|` for the exact values.
//!
//! Measured in a float dtype, the test was about something else. An int64
//! past 2^53 became the float64 beside it, and `2**60 + 1` was close to
//! `2**60.0` with no tolerance at all, though `==` said otherwise; a float32
//! pair measured against tolerances narrowed to float32, a wider band than the
//! one asked for; and two integers ignored the tolerances altogether, so
//! `100000` and `100001` were not close where `1 <= 1e-8 + 1e-5 * 100001`.
//! Each operand is now the number it holds, the tolerances are the float64s
//! given, and the comparison is decided in float64 with a bound on its
//! rounding, or exactly where the bound cannot decide it.

use std::cmp::Ordering;

use super::rounded_once::{Exact, Operand, wide_operand};
use crate::{
    error::{MinitensorError, Result},
    ops::kernels::broadcast_binary_data,
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// The tolerances of one comparison, as given.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tolerance {
    pub(crate) rtol: f64,
    pub(crate) atol: f64,
    pub(crate) equal_nan: bool,
}

/// An absolute slack covering a product of the tolerance and an operand that
/// underflows.
const TINY: f64 = f64::from_bits((1023 - 1000) << 52);

/// Whether `a` is close to `b`.
///
/// Equal values are close, equal infinities and signed zeros among them; a
/// NaN is close only to a NaN and only with `equal_nan`; an infinity is close
/// to nothing else. Otherwise the difference and the allowance are each
/// within `5u` of the scale `|high| + |low| + allowed` of what they are, for
/// the unit roundoff `u`, and a margin of twice that decides all but a
/// comparison on the edge, which is settled exactly.
#[inline(always)]
fn close<A: Operand, B: Operand>(a: A, b: B, tolerance: Tolerance) -> bool {
    let ((a_high, a_low), (b_high, b_low)) = (a.split(), b.split());
    if a_high == b_high && a_low == b_low {
        return true;
    }
    if a_high.is_nan() || b_high.is_nan() {
        return tolerance.equal_nan && a_high.is_nan() && b_high.is_nan();
    }
    if a_high.is_infinite() || b_high.is_infinite() {
        return false;
    }
    let (high, low) = (a_high - b_high, a_low - b_low);
    let difference = (high + low).abs();
    let allowed = tolerance.atol + tolerance.rtol * (b_high + b_low).abs();
    // An overflow makes the margin infinite and leaves both tests false.
    let margin = (high.abs() + low.abs() + allowed) * 2f64.powi(-50) + TINY;
    if difference + margin < allowed {
        return true;
    }
    if difference - margin > allowed {
        return false;
    }
    exactly_close(a.exact(), b.exact(), tolerance)
}

/// [`close`] for finite operands, in exact arithmetic.
#[cold]
#[inline(never)]
fn exactly_close(a: Option<Exact>, b: Option<Exact>, tolerance: Tolerance) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    let given = |value: f64| Exact::float(value).expect("the tolerances are finite");
    let (rtol, atol) = (given(tolerance.rtol), given(tolerance.atol));
    let term =
        |value: Exact, negative: bool| (negative, u128::from(value.magnitude), value.exponent);
    // `|a - b|` is `a - b` when `a` is the larger, and `b - a` otherwise.
    let larger = sign_of_sum(&[term(a, a.negative), term(b, !b.negative)]) == Ordering::Greater;
    let scaled = (
        false,
        u128::from(rtol.magnitude) * u128::from(b.magnitude),
        rtol.exponent + b.exponent,
    );
    // `atol + rtol · |b| - |a - b|`, not negative.
    let room = sign_of_sum(&[
        term(atol, false),
        scaled,
        term(a, a.negative != larger),
        term(b, b.negative == larger),
    ]);
    room != Ordering::Less
}

/// The sign of `Σ ±magnitude · 2^exponent`, exactly: the terms are laid
/// into whole 64-bit limbs from the lowest exponent up, the positive ones
/// and the negative ones apart, and the two totals compared.
fn sign_of_sum(terms: &[(bool, u128, i32)]) -> Ordering {
    let present = || terms.iter().filter(|term| term.1 != 0);
    let Some(lowest) = present().map(|term| term.2).min() else {
        return Ordering::Equal;
    };
    let width = present()
        .map(|term| (term.2 - lowest) as usize + 128)
        .max()
        .unwrap_or(0);
    // Two limbs past the widest term hold every carry of a handful of terms.
    let limbs = width / 64 + 2;
    let (mut positive, mut negative) = (vec![0u64; limbs], vec![0u64; limbs]);
    for &(is_negative, magnitude, exponent) in present() {
        let total = if is_negative {
            &mut negative
        } else {
            &mut positive
        };
        add_shifted(total, magnitude, (exponent - lowest) as usize);
    }
    positive.iter().rev().cmp(negative.iter().rev())
}

/// `total += magnitude << shift`, in little-endian 64-bit limbs.
fn add_shifted(total: &mut [u64], magnitude: u128, shift: usize) {
    let (word, bit) = (shift / 64, (shift % 64) as u32);
    let shifted = magnitude << bit;
    let spill = if bit == 0 {
        0
    } else {
        (magnitude >> (128 - bit)) as u64
    };
    let parts = [shifted as u64, (shifted >> 64) as u64, spill];
    let mut carry = false;
    let mut index = word;
    for part in parts {
        let (sum, first) = total[index].overflowing_add(part);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        total[index] = sum;
        carry = first || second;
        index += 1;
    }
    while carry {
        let (sum, overflow) = total[index].overflowing_add(1);
        total[index] = sum;
        carry = overflow;
        index += 1;
    }
}

/// `isclose` of `lhs` and `rhs` broadcast to `output_shape`, as booleans.
///
/// A float32 pair is read as it is; every other pair as int64 where an
/// operand is one and as float64 where it is not, which is exact either way,
/// so that five kernels serve every pairing.
pub(crate) fn isclose_data(
    lhs: &Tensor,
    rhs: &Tensor,
    tolerance: Tolerance,
    output_shape: &Shape,
) -> Result<TensorData> {
    macro_rules! slice {
        ($tensor:expr, $accessor:ident) => {
            $tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "isclose: failed to get the slice for ",
                    stringify!($accessor)
                ))
            })?
        };
    }
    macro_rules! kernel {
        ($lhs:expr, $rhs:expr, $left:ident, $right:ident) => {
            broadcast_binary_data(
                $lhs,
                $rhs,
                slice!($lhs, $left),
                slice!($rhs, $right),
                output_shape,
                |a, b| close(a, b, tolerance),
            )
        };
    }
    if lhs.dtype() == DataType::Float32 && rhs.dtype() == DataType::Float32 {
        return kernel!(lhs, rhs, as_f32_slice, as_f32_slice);
    }
    let (lhs, rhs) = (wide_operand(lhs)?, wide_operand(rhs)?);
    let (lhs, rhs) = (lhs.as_ref(), rhs.as_ref());
    match (lhs.dtype(), rhs.dtype()) {
        (DataType::Float64, DataType::Float64) => kernel!(lhs, rhs, as_f64_slice, as_f64_slice),
        (DataType::Int64, DataType::Float64) => kernel!(lhs, rhs, as_i64_slice, as_f64_slice),
        (DataType::Float64, DataType::Int64) => kernel!(lhs, rhs, as_f64_slice, as_i64_slice),
        (DataType::Int64, DataType::Int64) => kernel!(lhs, rhs, as_i64_slice, as_i64_slice),
        (left, right) => Err(MinitensorError::internal_error(format!(
            "isclose has no kernel for {left:?} beside {right:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance(rtol: f64, atol: f64) -> Tolerance {
        Tolerance {
            rtol,
            atol,
            equal_nan: false,
        }
    }

    #[test]
    fn the_exact_values_are_compared() {
        let none = tolerance(0.0, 0.0);
        assert!(!close((1_i64 << 60) + 1, 2f64.powi(60), none));
        assert!(close(1_i64 << 60, 2f64.powi(60), none));
        assert!(close(100_000_i64, 100_001_i64, tolerance(1e-5, 1e-8)));
        assert!(!close(100_000_i64, 100_002_i64, tolerance(1e-5, 1e-8)));
        assert!(close(5_i64, 6_i64, tolerance(0.0, 2.0)));
        // On the edge: `|1 - 1.1|` is `0.10000000000000009`, past `0.1`.
        assert!(!close(1.0f64, 1.1f64, tolerance(0.0, 0.1)));
        assert!(close(1.0f64, 1.1f64, tolerance(0.0, 0.10000000000000009)));
        // float32 operands against the float64 tolerance, not a float32 one.
        let gap = f64::from(1.1f32) - 1.0;
        assert!(!close(
            1.0f32,
            1.1f32,
            tolerance(0.0, gap - f64::EPSILON / 16.0)
        ));
        assert!(close(1.0f32, 1.1f32, tolerance(0.0, gap)));
        // Opposite huge values, whose difference overflows float64.
        assert!(!close(f64::MAX, -f64::MAX, tolerance(1.0, 0.0)));
        assert!(close(f64::MAX, -f64::MAX, tolerance(2.0, 0.0)));
        assert!(close(i64::MIN, i64::MIN, none));
        assert!(!close(i64::MIN, i64::MIN + 1, none));
    }

    #[test]
    fn the_float_test_agrees_with_the_exact_one() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let b = f64::from_bits(next() >> (1 + next() % 3));
            let tolerances = [
                tolerance(f64::from_bits(next() >> 2), 0.0),
                tolerance(0.0, f64::from_bits(next() >> 2)),
                tolerance(1e-5, 1e-8),
            ];
            for tolerance in tolerances {
                let allowed = tolerance.atol + tolerance.rtol * b.abs();
                // Just inside, on, and just past the edge.
                for a in [b + allowed, b - allowed, b + allowed * (1.0 + f64::EPSILON)] {
                    if !(a.is_finite() && b.is_finite() && allowed.is_finite()) {
                        continue;
                    }
                    assert_eq!(
                        close(a, b, tolerance),
                        exactly_close(a.exact(), b.exact(), tolerance),
                        "{a:e} {b:e} {tolerance:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn signs_of_sums_are_exact() {
        let one = (false, 1u128, 0);
        let tiny = (false, 1u128, -2000);
        let huge = (true, 1u128 << 100, 900);
        assert_eq!(
            sign_of_sum(&[huge, (false, 1u128 << 100, 900), tiny]),
            Ordering::Greater
        );
        assert_eq!(sign_of_sum(&[one, (true, 1, 0)]), Ordering::Equal);
        assert_eq!(
            sign_of_sum(&[one, (true, 1, 0), (true, 1, -1074)]),
            Ordering::Less
        );
        assert_eq!(sign_of_sum(&[]), Ordering::Equal);
    }
}
