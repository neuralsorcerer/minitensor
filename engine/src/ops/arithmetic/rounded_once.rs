// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `+`, `-`, `*` and `/` between an integer and a float, rounded once.
//!
//! Promoted the ordinary way, the integer is first rounded to the result's
//! float dtype and the operation then rounds again. Each step is within half
//! an ulp; the two together are not. `16777217 + 16777218.0` in float32 came
//! out `33554432` rather than `33554436`, a third of int32 quotients were an
//! ulp off, and an int64 timestamp past 2^53 divided by a float64 lost the
//! same way. The kernels here round the exact result once.
//!
//! Every int32, float32 and bool, and every int64 within 2^53, is exact in
//! float64. There the operation rounds once to 53 bits, which is the answer
//! for a float64 result. For a float32 one it is rounded to odd instead: the
//! operation's exact error -- a two-sum's, a split product's, a quotient's
//! remainder -- says which side of the float64 the exact result lies, and an
//! inexact result is moved to the odd one of the two float64s around it.
//! Narrowing that to float32 is then the exact result rounded once, since
//! the odd last bit is two places below where float32 rounds and stands for
//! everything the float64 could not hold. No branch is taken, so the loop
//! vectorizes as the plain kernels do.
//!
//! An int64 past 2^53 is carried out exactly in integer arithmetic instead,
//! element by element, and rounded from there.

use std::borrow::Cow;
use std::mem::MaybeUninit;

use crate::{
    error::{MinitensorError, Result},
    ops::kernels::{broadcast_binary_blocks, broadcast_binary_data, floored_quotient_f64},
    ops::map::BINARY_PAR_THRESHOLD,
    tensor::{DataType, Shape, Tensor, TensorData, TensorElement},
};

/// The operations carried here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

impl Arith {
    #[inline(always)]
    fn apply(self, x: f64, y: f64) -> f64 {
        match self {
            Arith::Add => x + y,
            Arith::Sub => x - y,
            Arith::Mul => x * y,
            Arith::Div => x / y,
        }
    }

    /// `x op y` in float64 for operands float64 holds exactly, rounded to
    /// odd: the result if exact, and otherwise whichever of the two float64s
    /// around the exact result has an odd last bit.
    #[inline(always)]
    fn to_odd(self, x: f64, y: f64) -> f64 {
        // The value, and a number whose sign is that of the exact result
        // less the value: zero when exact, and NaN, which moves nothing,
        // wherever an infinity or NaN is involved.
        let (value, rest) = match self {
            Arith::Add => two_sum(x, y),
            Arith::Sub => two_sum(x, -y),
            Arith::Mul => {
                let product = x * y;
                (product, product_error(x, y, product))
            }
            Arith::Div => {
                let quotient = x / y;
                // `x - quotient * y`, whose sign is the exact quotient's
                // side of `quotient` once the divisor's sign is taken out.
                // `x` and the product's leading part are within a factor of
                // two, so their difference is exact, and its rounding with
                // the product's error keeps the sign of the exact remainder.
                let product = quotient * y;
                let remainder = (x - product) - product_error(quotient, y, product);
                let rest = if y.is_sign_negative() {
                    -remainder
                } else {
                    remainder
                };
                (quotient, rest)
            }
        };
        odd(value, rest)
    }
}

/// `value` rounded to odd, given a number whose sign is that of the exact
/// result less `value`: zero when `value` is exact, and NaN to leave it be.
#[inline(always)]
fn odd(value: f64, rest: f64) -> f64 {
    let bits = value.to_bits();
    // Towards the exact result: a larger magnitude when it lies beyond the
    // value, a smaller one when it lies between the value and zero.
    let step = if (rest > 0.0) == (value > 0.0) {
        1
    } else {
        u64::MAX
    };
    let inexact_even = (rest != 0.0) & rest.is_finite() & (bits & 1 == 0);
    f64::from_bits(if inexact_even {
        bits.wrapping_add(step)
    } else {
        bits
    })
}

/// `a + b + c` rounded once to the nearest float64.
///
/// Two-sums take the exact sum apart into a leading float64 and two smaller
/// pieces; the pieces' sum rounded to odd keeps, two places below the
/// leading one's last bit, whether anything lies beyond it, so adding it to
/// the leading one rounds as the exact sum does.
#[inline(always)]
fn sum_of_three(a: f64, b: f64, c: f64) -> f64 {
    let (upper, upper_error) = two_sum(b, c);
    let (total, total_error) = two_sum(a, upper);
    let (rest, rest_error) = two_sum(total_error, upper_error);
    total + odd(rest, rest_error)
}

/// `x + y` and its exact rounding error.
#[inline(always)]
fn two_sum(x: f64, y: f64) -> (f64, f64) {
    let sum = x + y;
    let virtual_y = sum - x;
    let virtual_x = sum - virtual_y;
    (sum, (x - virtual_x) + (y - virtual_y))
}

/// The exact error of `product`, the float64 nearest `x * y`, from both
/// operands split into halves whose products float64 holds exactly.
#[inline(always)]
fn product_error(x: f64, y: f64, product: f64) -> f64 {
    let split = |value: f64| {
        let scaled = 134_217_729.0 * value;
        let high = scaled - (scaled - value);
        (high, value - high)
    };
    let ((x_high, x_low), (y_high, y_low)) = (split(x), split(y));
    ((x_high * y_high - product) + x_high * y_low + x_low * y_high) + x_low * y_low
}

/// Whether `lhs op rhs` promoted the ordinary way would round an integer
/// operand to `result` before the operation rounds it again.
///
/// A float32 result rounds every int32 and int64 above 2^24; a float64 one
/// rounds an int64 above 2^53, and holds every int32 exactly. A bool is 0 or 1
/// and exact in both.
pub(crate) fn rounds_twice(lhs: DataType, rhs: DataType, result: DataType) -> bool {
    let integral = |dtype: DataType| matches!(dtype, DataType::Int32 | DataType::Int64);
    match result {
        DataType::Float32 => integral(lhs) || integral(rhs),
        DataType::Float64 => lhs == DataType::Int64 || rhs == DataType::Int64,
        _ => false,
    }
}

/// `lhs op rhs` broadcast to `output_shape`, each element the exact result
/// rounded once to `result`. The operands are as given, not promoted; a pair
/// for which [`rounds_twice`] is false is refused.
pub(crate) fn rounded_once(
    lhs: &Tensor,
    rhs: &Tensor,
    op: Arith,
    result: DataType,
    output_shape: &Shape,
) -> Result<TensorData> {
    macro_rules! slice {
        ($tensor:expr, $accessor:ident) => {
            $tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get the slice for ",
                    stringify!($accessor)
                ))
            })?
        };
    }
    // An int32 beside a float32 is the pairing that turns up, so it has
    // kernels of its own, read in place with the operation fixed.
    match (result, lhs.dtype(), rhs.dtype()) {
        (DataType::Float32, DataType::Int32, DataType::Float32) => {
            let (left, right) = (slice!(lhs, as_i32_slice), slice!(rhs, as_f32_slice));
            return run::<f32, _, _>(lhs, rhs, left, right, op, output_shape);
        }
        (DataType::Float32, DataType::Float32, DataType::Int32) => {
            let (left, right) = (slice!(lhs, as_f32_slice), slice!(rhs, as_i32_slice));
            return run::<f32, _, _>(lhs, rhs, left, right, op, output_shape);
        }
        _ => {}
    }
    // Every other pairing is read as int64 where an operand is one and as
    // float64 where it is not -- exact either way -- so that a few kernels
    // serve them all. One per dtype pair and operation was a megabyte of the
    // extension, nearly all of it for pairings that are rare.
    let (lhs, rhs) = (wide_operand(lhs)?, wide_operand(rhs)?);
    let (lhs, rhs) = (lhs.as_ref(), rhs.as_ref());
    match (result, lhs.dtype(), rhs.dtype()) {
        // Integers, and an integer beside a bool, reach a float32 only by
        // dividing, so this pairing has the one kernel.
        (DataType::Float32, DataType::Float64, DataType::Float64) if op == Arith::Div => {
            let (left, right) = (slice!(lhs, as_f64_slice), slice!(rhs, as_f64_slice));
            broadcast_binary_data(lhs, rhs, left, right, output_shape, |a: f64, b: f64| {
                once::<f32, f64, f64>(Arith::Div, a, b)
            })
        }
        (DataType::Float32, DataType::Int64, DataType::Float64) => {
            let (left, right) = (slice!(lhs, as_i64_slice), slice!(rhs, as_f64_slice));
            run_any::<f32, _, _>(lhs, rhs, left, right, op, output_shape)
        }
        (DataType::Float32, DataType::Float64, DataType::Int64) => {
            let (left, right) = (slice!(lhs, as_f64_slice), slice!(rhs, as_i64_slice));
            run_any::<f32, _, _>(lhs, rhs, left, right, op, output_shape)
        }
        (DataType::Float64, DataType::Int64, DataType::Float64) => {
            let (left, right) = (slice!(lhs, as_i64_slice), slice!(rhs, as_f64_slice));
            run_any::<f64, _, _>(lhs, rhs, left, right, op, output_shape)
        }
        (DataType::Float64, DataType::Float64, DataType::Int64) => {
            let (left, right) = (slice!(lhs, as_f64_slice), slice!(rhs, as_i64_slice));
            run_any::<f64, _, _>(lhs, rhs, left, right, op, output_shape)
        }
        (DataType::Float64, DataType::Int64, DataType::Int64) => {
            let (left, right) = (slice!(lhs, as_i64_slice), slice!(rhs, as_i64_slice));
            run_any::<f64, _, _>(lhs, rhs, left, right, op, output_shape)
        }
        (result, left, right) => Err(MinitensorError::internal_error(format!(
            "{left:?} {op:?} {right:?} into {result:?} has no exactly rounded kernel"
        ))),
    }
}

/// An operand as the shared kernels read it: an int64 or a float64 as it is,
/// and anything else -- an int32, a float32, a bool -- in float64, which
/// holds each exactly.
fn wide_operand(tensor: &Tensor) -> Result<Cow<'_, Tensor>> {
    Ok(match tensor.dtype() {
        DataType::Int64 | DataType::Float64 => Cow::Borrowed(tensor),
        _ => Cow::Owned(tensor.astype(DataType::Float64)?),
    })
}

/// [`run`] with the operation chosen per element, for the pairings beside an
/// int64, whose loops branch on each element's size in any case. One plain
/// loop over blocks serves them: the vectorized builds the element-wise maps
/// keep for every closure would buy a branching loop nothing.
fn run_any<T: Target, A: Operand, B: Operand>(
    lhs: &Tensor,
    rhs: &Tensor,
    left: &[A],
    right: &[B],
    op: Arith,
    output_shape: &Shape,
) -> Result<TensorData> {
    scalar_blocks(lhs, rhs, left, right, output_shape, move |a: A, b: B| {
        once::<T, A, B>(op, a, b)
    })
}

/// `element` over the broadcast operands, block by block, in one plain loop.
fn scalar_blocks<T: Target, A: Operand, B: Operand>(
    lhs: &Tensor,
    rhs: &Tensor,
    left: &[A],
    right: &[B],
    output_shape: &Shape,
    element: impl Fn(A, B) -> T + Send + Sync,
) -> Result<TensorData> {
    // SAFETY: the block closure writes every element of the block it is
    // handed.
    let out = unsafe {
        broadcast_binary_blocks(
            left,
            right,
            lhs.shape(),
            rhs.shape(),
            output_shape,
            BINARY_PAR_THRESHOLD,
            |a: &[A], b: &[B], out: &mut [MaybeUninit<T>]| {
                for ((slot, &x), &y) in out.iter_mut().zip(a).zip(b) {
                    slot.write(element(x, y));
                }
            },
        )
    };
    Ok(TensorData::from_vec(out, T::DTYPE, lhs.device()))
}

/// The kernel for one pair of element types, with the operation fixed at
/// compile time so the loop's body is the operation's arithmetic alone.
fn run<T: Target, A: Operand, B: Operand>(
    lhs: &Tensor,
    rhs: &Tensor,
    left: &[A],
    right: &[B],
    op: Arith,
    output_shape: &Shape,
) -> Result<TensorData> {
    macro_rules! with {
        ($op:expr) => {
            broadcast_binary_data(lhs, rhs, left, right, output_shape, |a: A, b: B| {
                once::<T, A, B>($op, a, b)
            })
        };
    }
    match op {
        Arith::Add => with!(Arith::Add),
        Arith::Sub => with!(Arith::Sub),
        Arith::Mul => with!(Arith::Mul),
        Arith::Div => with!(Arith::Div),
    }
}

/// One element: from float64 where float64 holds both operands, and exactly
/// where it does not. For element types float64 always holds, the exact path
/// is not compiled into the loop at all.
#[inline(always)]
fn once<T: Target, A: Operand, B: Operand>(op: Arith, a: A, b: B) -> T {
    let (x, y) = (a.wide(), b.wide());
    if a.fits() && b.fits() {
        return T::from_wide(op, x, y);
    }
    if matches!(op, Arith::Add | Arith::Sub)
        && let Some(sum) = T::sum_wide(op, a.split(), b.split())
    {
        return sum;
    }
    exactly::<T>(op, a.exact(), b.exact(), op.apply(x, y))
}

/// A finite value as `±magnitude · 2^exponent`.
#[derive(Clone, Copy, Debug)]
struct Exact {
    negative: bool,
    magnitude: u64,
    exponent: i32,
}

impl Exact {
    fn integer(value: i64) -> Self {
        Self {
            negative: value < 0,
            magnitude: value.unsigned_abs(),
            exponent: 0,
        }
    }

    fn float(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let field = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1 << 52) - 1);
        let (magnitude, exponent) = if field == 0 {
            (fraction, -1074)
        } else {
            (fraction | 1 << 52, field - 1075)
        };
        Some(Self {
            negative: bits >> 63 == 1,
            magnitude,
            exponent,
        })
    }

    fn negated(self) -> Self {
        Self {
            negative: !self.negative,
            ..self
        }
    }
}

/// An element type an operand can have.
trait Operand: Copy + Send + Sync {
    /// Whether float64 holds every value of the type.
    const WIDE: bool;
    /// The value in float64.
    fn wide(self) -> f64;
    /// Whether that is all of it.
    #[inline(always)]
    fn fits(self) -> bool {
        Self::WIDE
    }
    /// The value exactly; `None` for an infinity or a NaN.
    fn exact(self) -> Option<Exact>;
    /// The value as the float64 nearest it and the exact remainder, which
    /// float64 also holds.
    #[inline(always)]
    fn split(self) -> (f64, f64) {
        (self.wide(), 0.0)
    }
}

impl Operand for bool {
    const WIDE: bool = true;
    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(u8::from(self))
    }
    fn exact(self) -> Option<Exact> {
        Some(Exact::integer(i64::from(self)))
    }
}

impl Operand for i32 {
    const WIDE: bool = true;
    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(self)
    }
    fn exact(self) -> Option<Exact> {
        Some(Exact::integer(i64::from(self)))
    }
}

impl Operand for i64 {
    const WIDE: bool = false;
    #[inline(always)]
    fn wide(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn fits(self) -> bool {
        self.unsigned_abs() <= 1 << 53
    }
    fn exact(self) -> Option<Exact> {
        Some(Exact::integer(self))
    }
    #[inline(always)]
    fn split(self) -> (f64, f64) {
        // The nearest float64 is within 2^10 of any int64. It can be 2^63,
        // which converts back as `i64::MAX`, one short.
        let nearest = self as f64;
        let short = if nearest >= 2f64.powi(63) { 1.0 } else { 0.0 };
        (nearest, self.wrapping_sub(nearest as i64) as f64 - short)
    }
}

impl Operand for f32 {
    const WIDE: bool = true;
    #[inline(always)]
    fn wide(self) -> f64 {
        f64::from(self)
    }
    fn exact(self) -> Option<Exact> {
        Exact::float(f64::from(self))
    }
}

impl Operand for f64 {
    const WIDE: bool = true;
    #[inline(always)]
    fn wide(self) -> f64 {
        self
    }
    fn exact(self) -> Option<Exact> {
        Exact::float(self)
    }
}

/// A float dtype a result is rounded to.
trait Target: TensorElement {
    /// Significand bits, the leading one included.
    const DIGITS: i32;
    /// The exponent of the smallest subnormal.
    const LEAST: i32;
    /// The exponent of the largest finite value's leading bit.
    const GREATEST: i32;
    /// The exponent of the smallest normal value.
    const LEAST_NORMAL: i32;
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn narrow(wide: f64) -> Self;
    /// `x op y`, rounded once, for operands float64 holds exactly.
    fn from_wide(op: Arith, x: f64, y: f64) -> Self;
    /// `x op y` for a sum or difference of operands split by
    /// [`Operand::split`], rounded once; `None` to leave it to the exact path.
    fn sum_wide(op: Arith, x: (f64, f64), y: (f64, f64)) -> Option<Self>;
    /// The value with these sign, biased exponent and fraction fields.
    fn assemble(negative: bool, field: u64, fraction: u64) -> Self;
    /// `±bits · 2^exponent`, rounded, for a result that is normal however
    /// `bits` rounds.
    fn scaled(negative: bool, bits: u64, exponent: i32) -> Self;
}

impl Target for f32 {
    const DIGITS: i32 = 24;
    const LEAST: i32 = -149;
    const GREATEST: i32 = 127;
    const LEAST_NORMAL: i32 = -126;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;

    #[inline(always)]
    fn narrow(wide: f64) -> Self {
        wide as f32
    }

    #[inline(always)]
    fn from_wide(op: Arith, x: f64, y: f64) -> Self {
        op.to_odd(x, y) as f32
    }

    #[inline(always)]
    fn sum_wide(_op: Arith, _x: (f64, f64), _y: (f64, f64)) -> Option<Self> {
        None
    }

    fn assemble(negative: bool, field: u64, fraction: u64) -> Self {
        f32::from_bits((u32::from(negative) << 31) | ((field as u32) << 23) | fraction as u32)
    }

    fn scaled(negative: bool, bits: u64, exponent: i32) -> Self {
        let unscaled = (bits as f32).to_bits() as i32;
        f32::from_bits((unscaled + (exponent << 23)) as u32 | (u32::from(negative) << 31))
    }
}

impl Target for f64 {
    const DIGITS: i32 = 53;
    const LEAST: i32 = -1074;
    const GREATEST: i32 = 1023;
    const LEAST_NORMAL: i32 = -1022;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;

    #[inline(always)]
    fn narrow(wide: f64) -> Self {
        wide
    }

    #[inline(always)]
    fn from_wide(op: Arith, x: f64, y: f64) -> Self {
        op.apply(x, y)
    }

    #[inline(always)]
    fn sum_wide(op: Arith, x: (f64, f64), y: (f64, f64)) -> Option<Self> {
        let y = match op {
            Arith::Add => y,
            Arith::Sub => (-y.0, -y.1),
            Arith::Mul | Arith::Div => return None,
        };
        // Only one operand of a sum reaching a float result can be an int64
        // past 2^53, so one remainder is zero. An infinity or NaN, and a sum
        // that could overflow, go to the exact path.
        let large = |value: f64| value.is_nan() || value.abs() >= 2f64.powi(1000);
        if large(x.0) || large(y.0) {
            return None;
        }
        Some(sum_of_three(x.0, y.0, x.1 + y.1))
    }

    fn assemble(negative: bool, field: u64, fraction: u64) -> Self {
        f64::from_bits((u64::from(negative) << 63) | (field << 52) | fraction)
    }

    fn scaled(negative: bool, bits: u64, exponent: i32) -> Self {
        let unscaled = (bits as f64).to_bits() as i64;
        f64::from_bits(
            (unscaled + (i64::from(exponent) << 52)) as u64 | (u64::from(negative) << 63),
        )
    }
}

/// `a op b` from the exact operands, rounded once.
///
/// `plain` is the operation on the operands in float64. It is the answer
/// wherever the result is exact whatever the operands' rounding -- an
/// infinity or NaN operand, a zero product or quotient, a division by zero,
/// a sum that cancels to zero -- and it gives those their IEEE signs.
#[cold]
#[inline(never)]
fn exactly<T: Target>(op: Arith, a: Option<Exact>, b: Option<Exact>, plain: f64) -> T {
    let (Some(a), Some(b)) = (a, b) else {
        return T::narrow(plain);
    };
    match op {
        Arith::Add | Arith::Sub => {
            let b = if op == Arith::Sub { b.negated() } else { b };
            match (a.magnitude, b.magnitude) {
                (0, 0) => T::narrow(plain),
                (0, _) => rounded(b.negative, u128::from(b.magnitude), b.exponent),
                (_, 0) => rounded(a.negative, u128::from(a.magnitude), a.exponent),
                _ => match sum(a, b) {
                    Some((negative, magnitude, exponent)) => rounded(negative, magnitude, exponent),
                    None => T::narrow(plain),
                },
            }
        }
        Arith::Mul => {
            if a.magnitude == 0 || b.magnitude == 0 {
                return T::narrow(plain);
            }
            rounded(
                a.negative != b.negative,
                u128::from(a.magnitude) * u128::from(b.magnitude),
                a.exponent + b.exponent,
            )
        }
        Arith::Div => {
            if a.magnitude == 0 || b.magnitude == 0 {
                return T::narrow(plain);
            }
            // The dividend's leading bit moved to bit 126 leaves a quotient
            // of at least 63 bits, ten past float64's; a remainder is marked
            // in the last of them, far below where the quotient rounds.
            let shift = a.magnitude.leading_zeros() + 63;
            let dividend = u128::from(a.magnitude) << shift;
            let divisor = u128::from(b.magnitude);
            let quotient = dividend / divisor;
            let inexact = u128::from(quotient * divisor != dividend);
            rounded(
                a.negative != b.negative,
                quotient | inexact,
                a.exponent - shift as i32 - b.exponent,
            )
        }
    }
}

/// `a + b` for two nonzero values as `±magnitude · 2^exponent`, or `None`
/// when they cancel exactly.
///
/// Each significand is moved up to bit 125, which leaves at least 62 zero
/// bits beneath it, so the two line up exactly unless their leading bits are
/// more than 62 apart. Then the smaller lies far below where the result
/// rounds, and all that matters of the bits shifted out is that some were:
/// that is marked in the last bit kept, which leaves the sum on the same side
/// of every halfway point as the exact sum.
fn sum(a: Exact, b: Exact) -> Option<(bool, u128, i32)> {
    let lift = |value: Exact| {
        let shift = value.magnitude.leading_zeros() + 62;
        (
            value.negative,
            u128::from(value.magnitude) << shift,
            value.exponent - shift as i32,
        )
    };
    let (a, b) = (lift(a), lift(b));
    let ((negative, larger, exponent), (other_negative, smaller, other_exponent)) =
        if (a.2, a.1) >= (b.2, b.1) {
            (a, b)
        } else {
            (b, a)
        };
    let apart = (exponent - other_exponent) as u32;
    let smaller = if apart >= 127 {
        1
    } else {
        let kept = smaller >> apart;
        kept | u128::from(kept << apart != smaller)
    };
    if negative == other_negative {
        Some((negative, larger + smaller, exponent))
    } else {
        let difference = larger - smaller;
        (difference != 0).then_some((negative, difference, exponent))
    }
}

/// `±magnitude · 2^exponent`, nonzero, rounded to the nearest `T`, ties to
/// even.
fn rounded<T: Target>(negative: bool, magnitude: u128, exponent: i32) -> T {
    let zeros = magnitude.leading_zeros();
    let leading = exponent + 127 - zeros as i32;
    if (T::LEAST_NORMAL..T::GREATEST).contains(&leading) {
        // Normal however it rounds. The leading 64 bits, with whether any
        // below them are set marked in the last, round as the whole does:
        // that bit is eleven or more places below where `T` rounds. The
        // conversion rounds to nearest, ties to even, and leaves a value in
        // [2^63, 2^64] for the exponent to be set on.
        let top = magnitude << zeros;
        let leading_bits = (top >> 64) as u64 | u64::from(top as u64 != 0);
        return T::scaled(negative, leading_bits, leading - 63);
    }
    let mut last = (leading - (T::DIGITS - 1)).max(T::LEAST);
    let dropped = last - exponent;
    let mut kept = if dropped <= 0 {
        magnitude << -dropped
    } else if dropped > 128 {
        // Below a quarter of the smallest subnormal.
        0
    } else {
        let (kept, rest) = if dropped == 128 {
            (0, magnitude)
        } else {
            (magnitude >> dropped, magnitude & ((1u128 << dropped) - 1))
        };
        let half = 1u128 << (dropped - 1);
        kept + u128::from(rest > half || (rest == half && kept & 1 == 1))
    };
    if kept == 1u128 << T::DIGITS {
        kept >>= 1;
        last += 1;
    }
    if kept >> (T::DIGITS - 1) == 0 {
        // Subnormal, or zero.
        return T::assemble(negative, 0, kept as u64);
    }
    let leading = last + T::DIGITS - 1;
    if leading > T::GREATEST {
        return if negative {
            T::NEG_INFINITY
        } else {
            T::INFINITY
        };
    }
    let bias = T::GREATEST;
    T::assemble(
        negative,
        (leading + bias) as u64,
        (kept as u64) & ((1u64 << (T::DIGITS - 1)) - 1),
    )
}

/// Floor division and its two remainders.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Modulus {
    /// `a // b`, the quotient rounded towards negative infinity.
    FloorDiv,
    /// `a % b`, which takes the divisor's sign.
    Floored,
    /// `fmod(a, b)`, which takes the dividend's sign.
    Truncated,
}

/// `lhs // rhs` or one of its remainders, as [`rounded_once`] is for the
/// arithmetic: each element the exact result rounded once to `result`.
pub(crate) fn modulus_once(
    lhs: &Tensor,
    rhs: &Tensor,
    kind: Modulus,
    result: DataType,
    output_shape: &Shape,
) -> Result<TensorData> {
    macro_rules! slice {
        ($tensor:expr, $accessor:ident) => {
            $tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get the slice for ",
                    stringify!($accessor)
                ))
            })?
        };
    }
    // As for the arithmetic: an int32 beside a float32 has kernels of its
    // own, and every other pairing is read as int64 and float64.
    match (result, lhs.dtype(), rhs.dtype()) {
        (DataType::Float32, DataType::Int32, DataType::Float32) => {
            let (left, right) = (slice!(lhs, as_i32_slice), slice!(rhs, as_f32_slice));
            return run_modulus::<f32, _, _>(lhs, rhs, left, right, kind, output_shape);
        }
        (DataType::Float32, DataType::Float32, DataType::Int32) => {
            let (left, right) = (slice!(lhs, as_f32_slice), slice!(rhs, as_i32_slice));
            return run_modulus::<f32, _, _>(lhs, rhs, left, right, kind, output_shape);
        }
        _ => {}
    }
    // An integer pair keeps an integer result here, so only an int64 beside
    // a float arrives.
    let (lhs, rhs) = (wide_operand(lhs)?, wide_operand(rhs)?);
    let (lhs, rhs) = (lhs.as_ref(), rhs.as_ref());
    match (result, lhs.dtype(), rhs.dtype()) {
        (DataType::Float32, DataType::Int64, DataType::Float64) => {
            let (left, right) = (slice!(lhs, as_i64_slice), slice!(rhs, as_f64_slice));
            run_modulus_any::<f32, _, _>(lhs, rhs, left, right, kind, output_shape)
        }
        (DataType::Float32, DataType::Float64, DataType::Int64) => {
            let (left, right) = (slice!(lhs, as_f64_slice), slice!(rhs, as_i64_slice));
            run_modulus_any::<f32, _, _>(lhs, rhs, left, right, kind, output_shape)
        }
        (DataType::Float64, DataType::Int64, DataType::Float64) => {
            let (left, right) = (slice!(lhs, as_i64_slice), slice!(rhs, as_f64_slice));
            run_modulus_any::<f64, _, _>(lhs, rhs, left, right, kind, output_shape)
        }
        (DataType::Float64, DataType::Float64, DataType::Int64) => {
            let (left, right) = (slice!(lhs, as_f64_slice), slice!(rhs, as_i64_slice));
            run_modulus_any::<f64, _, _>(lhs, rhs, left, right, kind, output_shape)
        }
        (result, left, right) => Err(MinitensorError::internal_error(format!(
            "{left:?} {kind:?} {right:?} into {result:?} has no exactly rounded kernel"
        ))),
    }
}

/// [`run_modulus`] element by element, for the pairings beside an int64: one
/// past 2^53 always takes the exact path, so each element takes whichever
/// path it needs.
fn run_modulus_any<T: Target, A: Operand, B: Operand>(
    lhs: &Tensor,
    rhs: &Tensor,
    left: &[A],
    right: &[B],
    kind: Modulus,
    output_shape: &Shape,
) -> Result<TensorData> {
    scalar_blocks(lhs, rhs, left, right, output_shape, move |a: A, b: B| {
        modulus_element::<T, A, B>(kind, a, b)
    })
}

/// [`run`] for floor division and the remainders.
///
/// The exact path is a call, and a call in the loop keeps the loop scalar, so
/// it is not taken element by element: each block is filled from float64
/// without a branch, and only a block in which some element could not be is
/// gone over again for those.
fn run_modulus<T: Target, A: Operand, B: Operand>(
    lhs: &Tensor,
    rhs: &Tensor,
    left: &[A],
    right: &[B],
    kind: Modulus,
    output_shape: &Shape,
) -> Result<TensorData> {
    macro_rules! with {
        ($kind:expr) => {
            // SAFETY: `modulus_block` writes every element of the block it
            // is handed.
            unsafe {
                broadcast_binary_blocks(
                    left,
                    right,
                    lhs.shape(),
                    rhs.shape(),
                    output_shape,
                    BINARY_PAR_THRESHOLD,
                    |a: &[A], b: &[B], out: &mut [MaybeUninit<T>]| {
                        modulus_block::<T, A, B>($kind, a, b, out)
                    },
                )
            }
        };
    }
    let out = match kind {
        Modulus::FloorDiv => with!(Modulus::FloorDiv),
        Modulus::Floored => with!(Modulus::Floored),
        Modulus::Truncated => with!(Modulus::Truncated),
    };
    Ok(TensorData::from_vec(out, T::DTYPE, lhs.device()))
}

/// One block of floor division or a remainder, every element written.
#[inline(always)]
fn modulus_block<T: Target, A: Operand, B: Operand>(
    kind: Modulus,
    left: &[A],
    right: &[B],
    out: &mut [MaybeUninit<T>],
) {
    let mut pending = false;
    for ((slot, &a), &b) in out.iter_mut().zip(left).zip(right) {
        let (value, settled) = modulus_wide::<T, A, B>(kind, a, b);
        slot.write(value);
        pending |= !settled;
    }
    if pending {
        for ((slot, &a), &b) in out.iter_mut().zip(left).zip(right) {
            if !modulus_wide::<T, A, B>(kind, a, b).1 {
                slot.write(modulus_exactly::<T>(
                    kind,
                    a.exact(),
                    b.exact(),
                    a.wide(),
                    b.wide(),
                ));
            }
        }
    }
}

/// One element of floor division or a remainder from float64, and whether
/// that is the answer.
///
/// With finite operands float64 holds exactly and a quotient below 2^52,
/// [`truncated_division`] gives the truncated quotient and remainder
/// exactly. The truncated remainder is then the answer rounded once; the
/// floored one is that, or that plus the divisor rounded once; and the floor
/// quotient is the truncated one, or one below it. Anything else is done by
/// [`modulus_exactly`].
#[inline(always)]
fn modulus_wide<T: Target, A: Operand, B: Operand>(kind: Modulus, a: A, b: B) -> (T, bool) {
    let (x, y) = (a.wide(), b.wide());
    let (quotient, rest) = truncated_division(x, y);
    let settled = a.fits()
        & b.fits()
        & x.is_finite()
        & y.is_finite()
        & (y != 0.0)
        & (quotient.abs() < 2f64.powi(52));
    // A remainder of the dividend's sign, and a divisor of the other.
    let crosses = (rest != 0.0) & ((x < 0.0) != (y < 0.0));
    let value = match kind {
        // A zero remainder takes the sign of the dividend here, and of the
        // divisor below; the subtraction gives +0 either way.
        Modulus::Truncated => T::narrow(if rest == 0.0 { rest.copysign(x) } else { rest }),
        Modulus::FloorDiv => T::narrow(if crosses { quotient - 1.0 } else { quotient }),
        Modulus::Floored => {
            let shifted = T::from_wide(Arith::Add, rest, y);
            if rest == 0.0 {
                T::narrow(rest.copysign(y))
            } else if crosses {
                shifted
            } else {
                T::narrow(rest)
            }
        }
    };
    (value, settled)
}

/// One element of floor division or a remainder, whichever path it takes.
#[inline(always)]
fn modulus_element<T: Target, A: Operand, B: Operand>(kind: Modulus, a: A, b: B) -> T {
    if a.fits()
        && b.fits()
        && let (value, true) = modulus_wide::<T, A, B>(kind, a, b)
    {
        return value;
    }
    modulus_exactly::<T>(kind, a.exact(), b.exact(), a.wide(), b.wide())
}

/// `trunc(x / y)` and `x - trunc(x / y) * y` for finite float64 operands and
/// a nonzero divisor, both exact while the quotient is below 2^52, and of no
/// use otherwise.
///
/// The rounded quotient never falls short of the truncated one -- that is a
/// float64, and rounding is monotone -- but it can round up onto the next
/// whole number. Then `x - q * y` comes out with the sign opposite to `x`'s,
/// and one divisor more mends both. Each step is exact: `q * y` is within a
/// factor of two of `x` unless `q` is zero, so `x` less its leading part
/// is; the product's error is exact from the split; and every remainder
/// formed is one float64 holds, the true one being smaller than both
/// operands and no finer than the finer of them.
#[inline(always)]
fn truncated_division(x: f64, y: f64) -> (f64, f64) {
    // Below 2^52, the conversion to an integer is the truncation, and one
    // instruction where `trunc` is a call; the sign puts back a negative
    // zero. Above 2^52 the result is not used.
    let quotient = ((x / y) as i64 as f64).copysign(x / y);
    let product = quotient * y;
    let rest = (x - product) - product_error(quotient, y, product);
    let over = (rest != 0.0) & ((rest < 0.0) != (x < 0.0));
    let step = if (x < 0.0) != (y < 0.0) { -1.0 } else { 1.0 };
    if over {
        (quotient - step, rest + y * step)
    } else {
        (quotient, rest)
    }
}

/// Floor division or a remainder from the exact operands, rounded once.
///
/// `x` and `y` are the operands in float64. A zero, an infinity or a NaN
/// gives a result their rounding does not reach -- a zero, NaN, an infinity,
/// 0 or -1 -- so the float64 rules answer for those, except the remainder by
/// an infinite divisor, which is the dividend itself.
#[cold]
#[inline(never)]
fn modulus_exactly<T: Target>(
    kind: Modulus,
    a: Option<Exact>,
    b: Option<Exact>,
    x: f64,
    y: f64,
) -> T {
    let plain = || match kind {
        Modulus::FloorDiv => floored_quotient_f64(x, y),
        Modulus::Truncated => x % y,
        Modulus::Floored => {
            let rest = x % y;
            if rest == 0.0 {
                rest.copysign(y)
            } else if (rest < 0.0) != (y < 0.0) {
                rest + y
            } else {
                rest
            }
        }
    };
    let Some(a) = a else {
        return T::narrow(plain());
    };
    if a.magnitude == 0 || y == 0.0 || y.is_nan() {
        return T::narrow(plain());
    }
    let Some(b) = b else {
        // An infinite divisor: the quotient is 0 or -1, and the dividend is
        // its own truncated remainder.
        return match kind {
            Modulus::FloorDiv => T::narrow(plain()),
            _ => finish_remainder(kind, a, None, y),
        };
    };
    match kind {
        Modulus::FloorDiv => {
            let negative = a.negative != b.negative;
            // A negative quotient that is not whole floors one further out.
            match narrow_quotient(a, b) {
                Some((whole, inexact)) => match whole + u128::from(negative && inexact) {
                    0 => T::assemble(negative, 0, 0),
                    floor => rounded(negative, floor, 0),
                },
                None => {
                    let (limbs, inexact) = whole_quotient(a, b);
                    rounded_limbs(negative, limbs, negative && inexact)
                }
            }
        }
        _ => finish_remainder(kind, truncated_remainder(a, b), Some(b), y),
    }
}

/// `floor(x / y)` for finite float32 operands, exactly, rounded once to
/// float32. For a quotient past 2^24, where float32's midpoints are whole
/// numbers, the floor of the rounded quotient is not always the rounded
/// floor: a floor exactly halfway between two float32s rounds to even, where
/// the quotient just above it rounds up.
pub(crate) fn exact_floor_quotient_f32(x: f32, y: f32) -> f32 {
    let (x, y) = (f64::from(x), f64::from(y));
    modulus_exactly::<f32>(Modulus::FloorDiv, Exact::float(x), Exact::float(y), x, y)
}

/// [`exact_floor_quotient_f32`] for float64, past 2^53.
pub(crate) fn exact_floor_quotient_f64(x: f64, y: f64) -> f64 {
    modulus_exactly::<f64>(Modulus::FloorDiv, Exact::float(x), Exact::float(y), x, y)
}

/// A remainder from the truncated one, `rest`, rounded once: `b` is the
/// divisor, or `None` for an infinite one, and `y` the divisor in float64.
fn finish_remainder<T: Target>(kind: Modulus, rest: Exact, b: Option<Exact>, y: f64) -> T {
    if rest.magnitude == 0 {
        let negative = match kind {
            Modulus::Floored => y.is_sign_negative(),
            _ => rest.negative,
        };
        return T::assemble(negative, 0, 0);
    }
    if kind == Modulus::Floored && rest.negative != y.is_sign_negative() {
        // The floored remainder is the truncated one plus the divisor, which
        // is smaller than the divisor and so does not cancel to zero.
        return match b.and_then(|b| sum(rest, b)) {
            Some((negative, magnitude, exponent)) => rounded(negative, magnitude, exponent),
            None => T::narrow(y),
        };
    }
    rounded(rest.negative, u128::from(rest.magnitude), rest.exponent)
}

/// `a - trunc(a / b) * b` exactly, for nonzero finite `a` and `b`.
///
/// It takes `a`'s sign and is smaller than both in magnitude, so a u64
/// holds it in units of the finer of their last bits: `a`'s units are
/// reduced modulo `b`'s, or `a` is left as it is when `b` is the larger.
fn truncated_remainder(a: Exact, b: Exact) -> Exact {
    let (magnitude, exponent) = if a.exponent >= b.exponent {
        let divisor = u128::from(b.magnitude);
        let scale = power_of_two_modulo((a.exponent - b.exponent) as u32, divisor);
        (
            (u128::from(a.magnitude) % divisor * scale % divisor) as u64,
            b.exponent,
        )
    } else {
        let up = (b.exponent - a.exponent) as u32;
        if 64 - b.magnitude.leading_zeros() + up > 64 {
            (a.magnitude, a.exponent)
        } else {
            (a.magnitude % (b.magnitude << up), a.exponent)
        }
    };
    Exact {
        negative: a.negative,
        magnitude,
        exponent,
    }
}

/// `2^exponent mod modulus`, for a modulus below 2^64.
fn power_of_two_modulo(mut exponent: u32, modulus: u128) -> u128 {
    let mut result = 1 % modulus;
    let mut base = 2 % modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

/// `floor(|a| / |b|)` for nonzero finite `a` and `b`, and whether the division
/// left a remainder, when the dividend in units of the divisor's last bit
/// fits in 127 bits; `None` otherwise.
fn narrow_quotient(a: Exact, b: Exact) -> Option<(u128, bool)> {
    let (dividend, divisor) = if a.exponent < b.exponent {
        let up = (b.exponent - a.exponent) as u32;
        if 64 - b.magnitude.leading_zeros() + up > 64 {
            // The divisor is the larger.
            return Some((0, true));
        }
        (u128::from(a.magnitude), u128::from(b.magnitude << up))
    } else {
        let shift = (a.exponent - b.exponent) as u32;
        if 64 - a.magnitude.leading_zeros() + shift > 127 {
            return None;
        }
        (u128::from(a.magnitude) << shift, u128::from(b.magnitude))
    };
    let whole = dividend / divisor;
    Some((whole, whole * divisor != dividend))
}

/// `floor(|a| / |b|)` for nonzero finite `a` and `b`, in base-2^64 limbs from
/// the least significant, and whether the division left a remainder, for a
/// dividend whose last bit is not finer than the divisor's -- the quotients
/// [`narrow_quotient`] leaves.
///
/// The quotient of an int64 by a tiny float runs past any fixed width, and
/// its last bits still decide how it rounds when it is a whole number
/// halfway between two floats; long division keeps all of them.
fn whole_quotient(a: Exact, b: Exact) -> (Vec<u64>, bool) {
    let shift = (a.exponent - b.exponent) as u32;
    let (whole, part) = ((shift / 64) as usize, shift % 64);
    let mut limbs = vec![0u64; whole + 2];
    limbs[whole] = a.magnitude << part;
    if part != 0 {
        limbs[whole + 1] = a.magnitude >> (64 - part);
    }
    let divisor = u128::from(b.magnitude);
    let mut remainder = 0u128;
    for limb in limbs.iter_mut().rev() {
        let current = (remainder << 64) | u128::from(*limb);
        let quotient = current / divisor;
        remainder = current - quotient * divisor;
        *limb = quotient as u64;
    }
    (limbs, remainder != 0)
}

/// The whole number in `limbs`, plus one when `extra`, rounded to `T`.
fn rounded_limbs<T: Target>(negative: bool, mut limbs: Vec<u64>, extra: bool) -> T {
    if extra {
        // The top limb is below 2^63, so the carry stops inside.
        for limb in &mut limbs {
            let (sum, carry) = limb.overflowing_add(1);
            *limb = sum;
            if !carry {
                break;
            }
        }
    }
    match limbs.iter().rposition(|&limb| limb != 0) {
        None => T::assemble(negative, 0, 0),
        Some(0) => rounded(negative, u128::from(limbs[0]), 0),
        Some(top) => {
            // At least 65 bits, with whether any below them are set marked
            // in the last: twelve or more places below where `T` rounds.
            let leading = (u128::from(limbs[top]) << 64) | u128::from(limbs[top - 1]);
            let below = limbs[..top - 1].iter().any(|&limb| limb != 0);
            rounded(negative, leading | u128::from(below), 64 * (top as i32 - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_of(op: Arith, a: impl Operand, b: impl Operand) -> f32 {
        once::<f32, _, _>(op, a, b)
    }

    #[test]
    fn a_halfway_float64_is_not_rounded_again() {
        // 16777217 is halfway between two float32s; the tiny addend puts
        // the sum above it.
        let tiny = (2.0f32).powi(-40);
        assert_eq!(f32_of(Arith::Add, 16_777_217_i32, tiny), 16_777_218.0);
        assert_eq!(f32_of(Arith::Sub, 16_777_217_i32, tiny), 16_777_216.0);
        assert_eq!(f32_of(Arith::Add, 16_777_217_i32, 0.0f32), 16_777_216.0);
        assert_eq!(
            f32_of(Arith::Add, 16_777_217_i32, 16_777_218.0f32),
            33_554_436.0
        );
    }

    #[test]
    fn integer_quotients_round_once() {
        // A tie, an ordinary quotient, and a negative one.
        assert_eq!(f32_of(Arith::Div, 16_777_217_i32, 1_i32), 16_777_216.0);
        assert_eq!(f32_of(Arith::Div, 2_147_483_647_i32, 3_i32), 715_827_904.0);
        assert_eq!(f32_of(Arith::Div, -16_777_219_i32, 7_i32), -2_396_745.5);
    }

    #[test]
    fn int64_past_2_pow_53_is_carried_exactly() {
        let big = (1_i64 << 60) + (1 << 36);
        // Halfway between two float32s, nudged up by the half.
        assert_eq!(
            f32_of(Arith::Add, big, 0.5f32),
            ((1u64 << 60) + (1 << 37)) as f32
        );
        let odd = (1_i64 << 53) + 1;
        assert_eq!(
            once::<f64, _, _>(Arith::Add, odd, 1.0f64),
            9_007_199_254_740_994.0
        );
        assert_eq!(
            once::<f64, _, _>(Arith::Mul, odd, 3.0f64),
            27_021_597_764_222_980.0
        );
        // 2^53 + 1 is three times an integer float64 holds; rounded to
        // 2^53 first, the quotient came out half below it.
        assert_eq!(
            once::<f64, _, _>(Arith::Div, odd, 3_i64),
            3_002_399_751_580_331.0
        );
        assert_eq!(
            once::<f64, _, _>(Arith::Sub, i64::MIN, 0.0f64),
            -(2f64.powi(63))
        );
    }

    #[test]
    fn specials_and_signed_zeros_follow_the_float_rules() {
        assert!(f32_of(Arith::Add, 3_i32, f32::NAN).is_nan());
        assert_eq!(f32_of(Arith::Div, 3_i32, 0_i32), f32::INFINITY);
        assert!(f32_of(Arith::Div, 0_i32, 0_i32).is_nan());
        assert_eq!(
            f32_of(Arith::Mul, 0_i32, -3.0f32).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            f32_of(Arith::Div, 0_i32, -3_i32).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            f32_of(Arith::Add, 0_i32, -0.0f32).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            f32_of(Arith::Sub, 5_i32, 5.0f32).to_bits(),
            0.0f32.to_bits()
        );
        let big = i64::MAX;
        assert_eq!(f32_of(Arith::Mul, big, f32::INFINITY), f32::INFINITY);
        assert_eq!(
            f32_of(Arith::Div, big, f32::NEG_INFINITY).to_bits(),
            (-0.0f32).to_bits()
        );
    }

    #[test]
    fn overflow_and_underflow_round_at_the_edges() {
        // Past float32's largest value, by a quotient.
        assert_eq!(f32_of(Arith::Div, 1_i32, f32::from_bits(1)), f32::INFINITY);
        // Into the subnormals.
        let smallest = f32::from_bits(1);
        assert_eq!(f32_of(Arith::Div, smallest, 2_i32), 0.0);
        assert_eq!(
            f32_of(Arith::Div, f32::from_bits(3), 2_i32),
            f32::from_bits(2)
        );
        assert_eq!(
            f32_of(Arith::Mul, f32::from_bits(3), 3_i32),
            f32::from_bits(9)
        );
        assert_eq!(
            f32_of(Arith::Div, f32::MIN_POSITIVE, 3_i32),
            f32::MIN_POSITIVE / 3.0
        );
    }

    /// A small xorshift generator, for the sweeps below.
    fn generator(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    #[test]
    fn exact_rounding_agrees_with_float64_where_float64_is_exact() {
        // Every path of `rounded`, checked against float64's own rounding
        // on values float64 holds.
        let mut next = generator(0x9e37_79b9_7f4a_7c15);
        for round in 0..40_000 {
            let a = f64::from_bits(next());
            // Every other pair close together, for cancellation and for
            // sums that line up without shifting anything out.
            let b = if round % 2 == 0 {
                f64::from_bits(next())
            } else {
                f64::from_bits(a.to_bits() ^ (next() & 0x8000_00ff_ffff_ffff))
            };
            if !(a.is_finite() && b.is_finite()) {
                continue;
            }
            for op in [Arith::Add, Arith::Sub, Arith::Mul, Arith::Div] {
                let plain = op.apply(a, b);
                let exact: f64 = exactly(op, Exact::float(a), Exact::float(b), plain);
                assert!(
                    exact.to_bits() == plain.to_bits() || (exact.is_nan() && plain.is_nan()),
                    "{a:e} {op:?} {b:e}: {exact:e} against {plain:e}"
                );
            }
        }
    }

    #[test]
    fn rounding_to_odd_agrees_with_exact_rounding() {
        // The float32 fast path against the exact one: integers of every
        // width beside float32s of every exponent, integer quotients, and
        // integers that are halfway between two float32s beside tiny ones.
        let mut next = generator(0x2545_f491_4f6c_dd1d);
        let float = |next: &mut dyn FnMut() -> u64| loop {
            let value = f32::from_bits(next() as u32);
            if value.is_finite() {
                return value;
            }
        };
        for round in 0..200_000 {
            let int = (next() as i32) >> (next() % 32);
            let (x, y) = match round % 3 {
                0 => (f64::from(int), f64::from(float(&mut next))),
                1 => (f64::from(int), f64::from((next() as i32) >> (next() % 32))),
                _ => {
                    let halfway = (((next() % (1 << 23)) as i64 | 1 << 23) * 2 + 1) as f64;
                    let tiny = f64::from(f32::from_bits(next() as u32 & 0x9fff_ffff));
                    (halfway, tiny)
                }
            };
            for op in [Arith::Add, Arith::Sub, Arith::Mul, Arith::Div] {
                let fast = <f32 as Target>::from_wide(op, x, y);
                let exact: f32 = exactly(op, Exact::float(x), Exact::float(y), op.apply(x, y));
                assert!(
                    fast.to_bits() == exact.to_bits() || (fast.is_nan() && exact.is_nan()),
                    "{x:e} {op:?} {y:e}: {fast:e} against {exact:e}"
                );
                let fast = <f32 as Target>::from_wide(op, y, x);
                let exact: f32 = exactly(op, Exact::float(y), Exact::float(x), op.apply(y, x));
                assert!(
                    fast.to_bits() == exact.to_bits() || (fast.is_nan() && exact.is_nan()),
                    "{y:e} {op:?} {x:e}: {fast:e} against {exact:e}"
                );
            }
        }
    }

    #[test]
    fn floor_division_and_remainders_agree_with_the_exact_path() {
        // Integers of every width beside floats of every exponent, in both
        // orders, through float64 and through the exact path.
        let mut next = generator(0xbb67_ae85_84ca_a73b);
        for _ in 0..100_000 {
            let int = ((next() as i64) >> (11 + next() % 52)) as f64;
            let float = loop {
                let value = f32::from_bits(next() as u32);
                if value.is_finite() && value != 0.0 {
                    break f64::from(value);
                }
            };
            for kind in [Modulus::FloorDiv, Modulus::Floored, Modulus::Truncated] {
                for (x, y) in [(int, float), (float, int)] {
                    let fast: f32 = modulus_element::<f32, f64, f64>(kind, x, y);
                    let exact: f32 =
                        modulus_exactly::<f32>(kind, Exact::float(x), Exact::float(y), x, y);
                    assert!(
                        fast.to_bits() == exact.to_bits() || (fast.is_nan() && exact.is_nan()),
                        "{x:e} {kind:?} {y:e}: {fast:e} against {exact:e}"
                    );
                    let fast: f64 = modulus_element::<f64, f64, f64>(kind, x, y);
                    let exact: f64 =
                        modulus_exactly::<f64>(kind, Exact::float(x), Exact::float(y), x, y);
                    assert!(
                        fast.to_bits() == exact.to_bits() || (fast.is_nan() && exact.is_nan()),
                        "{x:e} {kind:?} {y:e}: {fast:e} against {exact:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn int64_past_2_pow_53_divides_and_reduces_exactly() {
        let odd = (1_i64 << 60) + 1;
        // 2^60 is 1 more than a multiple of 3, so the remainder is 2.
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::Floored, odd, 3.0f64),
            2.0
        );
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::Truncated, -odd, 3.0f64),
            -2.0
        );
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::Floored, -odd, 3.0f64),
            1.0
        );
        assert_eq!(
            modulus_element::<f32, _, _>(Modulus::Floored, odd, 3.0f32),
            2.0
        );
        // floor((2^60 + 1) / 2) = 2^59 exactly; and below zero, one further.
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::FloorDiv, odd, 2.0f64),
            2f64.powi(59)
        );
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::FloorDiv, -odd, 2.0f64),
            -(2f64.powi(59)) - 1.0
        );
        // A divisor larger than the dividend, of the other sign.
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::FloorDiv, odd, -1e30f64),
            -1.0
        );
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::Floored, odd, -1e30f64),
            odd as f64 - 1e30
        );
        // An infinite divisor leaves the dividend, rounded once.
        assert_eq!(
            modulus_element::<f32, _, _>(
                Modulus::Truncated,
                (1_i64 << 60) + (1 << 36) + 1,
                f32::INFINITY
            ),
            ((1u64 << 60) + (1 << 37)) as f32
        );
        assert_eq!(
            modulus_element::<f32, _, _>(Modulus::Floored, -5_i64, f32::INFINITY),
            f32::INFINITY
        );
        assert!(modulus_element::<f64, _, _>(Modulus::Floored, odd, 0.0f64).is_nan());
        assert_eq!(
            modulus_element::<f64, _, _>(Modulus::FloorDiv, odd, 0.0f64),
            f64::INFINITY
        );
        // A quotient past 2^128, whole and halfway between two float32s,
        // rounds to even where its float64 quotient would round up.
        let tiny = f32::from_bits(1);
        let halfway = (1_i64 << 25) + 1;
        assert_eq!(
            modulus_element::<f32, _, _>(Modulus::FloorDiv, halfway, tiny),
            f32::INFINITY
        );
        let quotient: f32 = modulus_element(Modulus::FloorDiv, 16_777_217_i32, 2f32.powi(-100));
        assert_eq!(quotient, 2f32.powi(124));
        let quotient: f32 =
            modulus_element(Modulus::FloorDiv, 16_777_217_i32, 3f32 * 2f32.powi(-100));
        let exact = (16_777_217_u128 << 100) / 3;
        assert_eq!(quotient, exact as f32);
    }

    #[test]
    fn a_sum_of_three_rounds_as_the_exact_sum() {
        // int64s past 2^53 beside float64s of every size, close to their
        // negation, and halfway between two float64s the int64 lies near.
        let mut next = generator(0x6a09_e667_f3bc_c909);
        for round in 0..300_000 {
            let int = (next() as i64) >> (next() % 11);
            let near = int as f64;
            let y = match round % 4 {
                0 => f64::from_bits(next() >> 1),
                1 => -near + ((next() as i64) >> (40 + next() % 24)) as f64,
                2 => {
                    f64::from_bits((next() & 0x800f_ffff_ffff_ffff) | ((1023 + next() % 80) << 52))
                }
                _ => {
                    let spacing = 2f64.powi(near.abs().log2().floor() as i32 - 52);
                    let offset = (next() % 8) as f64 * 0.25 - 1.0;
                    spacing * (offset + f64::from_bits(next() >> 12 | 0x3cb0_0000_0000_0000))
                }
            };
            if !y.is_finite() {
                continue;
            }
            for op in [Arith::Add, Arith::Sub] {
                for swapped in [false, true] {
                    let ((a, ea), (b, eb)) = if swapped {
                        ((y.split(), y.exact()), (int.split(), int.exact()))
                    } else {
                        ((int.split(), int.exact()), (y.split(), y.exact()))
                    };
                    let Some(fast) = <f64 as Target>::sum_wide(op, a, b) else {
                        continue;
                    };
                    let exact: f64 = exactly(op, ea, eb, op.apply(a.0, b.0));
                    assert_eq!(
                        fast.to_bits(),
                        exact.to_bits(),
                        "{a:?} {op:?} {b:?}: {fast:e} against {exact:e}"
                    );
                }
            }
        }
    }
}
