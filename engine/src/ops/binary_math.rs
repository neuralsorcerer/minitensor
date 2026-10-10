// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Element-wise binary functions whose answer is a float.
//!
//! `atan2`, `hypot`, `copysign` and `xlogy` differ only in three kernels each
//! -- the function and its two partial derivatives -- so that is all each one
//! writes here. The promotion, the broadcast, the graph bookkeeping and the
//! backward pass are shared, and the derivative of an op sits next to the op.
//!
//! Promotion follows `/`: the answer is a float even when both operands are
//! integers, and float64 only when an operand already is.

use crate::{
    autograd::{FloatBinaryBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_and_broadcast},
    ops::kernels::broadcast_binary_arm,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// One element-wise binary partial derivative, in both float widths.
///
/// A pair of function pointers rather than a generic, so that a single value
/// can stand for both halves of a derivative in [`FloatBinaryBackward`],
/// which stores two of them and does not know which op it came from.
///
/// Forwards do *not* go through this, and the difference is larger than it
/// looks. A function pointer is opaque to the optimizer, so a per-element call
/// through one cannot inline: routed this way `fmax` ran at 0.666ms over a
/// million float32 where `ops::minmax`'s `maximum` -- the same shape of work
/// behind a closure, with a branchier body -- ran at 0.174. Forwards are
/// declared by [`float_kernel!`] instead, which names both halves at module
/// scope so [`float_binary_mono`] instantiates per op. That took `fmax` to
/// 0.195 and `copysign` from 0.56 to 0.15.
///
/// The backward keeps the pointers because it is not hot and because
/// monomorphizing it would mean a generic `FloatBinaryBackward` per op.
pub type FloatBinaryKernel = (fn(f32, f32) -> f32, fn(f64, f64) -> f64);

/// Defines a forward kernel as two named functions, one per width, from a
/// single body.
///
/// Named rather than hidden in a `const` so the call site passes a concrete
/// function *item* and the loop inlines it -- see [`FloatBinaryKernel`] for
/// what that is worth. The body has to typecheck as `f32` and as `f64`, which
/// rules out naming either type inside it -- write `a.hypot(b)`, not
/// `f64::hypot(a, b)`.
macro_rules! float_kernel {
    ($(#[$meta:meta])* $narrow:ident, $wide:ident, |$a:pat_param, $b:pat_param| $body:expr) => {
        $(#[$meta])*
        #[inline(always)]
        fn $narrow($a: f32, $b: f32) -> f32 {
            $body
        }

        #[doc = concat!("See [`", stringify!($narrow), "`].")]
        #[inline(always)]
        fn $wide($a: f64, $b: f64) -> f64 {
            $body
        }
    };
}

/// Defines one [`FloatBinaryKernel`] from a single body, instantiated at both
/// widths. Used for the partial derivatives, which are stored rather than
/// called directly. Same typechecking rule as [`float_kernel!`].
macro_rules! float_pair {
    ($(#[$meta:meta])* $vis:vis $name:ident, |$a:pat_param, $b:pat_param| $body:expr) => {
        $(#[$meta])*
        $vis const $name: FloatBinaryKernel = {
            #[inline(always)]
            fn narrow($a: f32, $b: f32) -> f32 {
                $body
            }
            #[inline(always)]
            fn wide($a: f64, $b: f64) -> f64 {
                $body
            }
            (narrow, wide)
        };
    };
}

// --- atan2 -----------------------------------------------------------------

float_pair!(
    /// `d/dy atan2(y, x) = x / (x^2 + y^2)`, grouped through `hypot` so the
    /// sum of squares cannot overflow for operands past the square root of
    /// the dtype's range. Its magnitude is at most `1 / hypot`, so it is 0
    /// at an infinite point, where `inf / inf` would read NaN.
    ATAN2_D_Y, |y, x| {
        let h = y.hypot(x);
        if h.is_infinite() { 0.0 } else { (x / h) / h }
    }
);
float_pair!(
    /// `d/dx atan2(y, x) = -y / (x^2 + y^2)`, grouped and bounded as in
    /// `ATAN2_D_Y`.
    ATAN2_D_X, |y, x| {
        let h = y.hypot(x);
        if h.is_infinite() { 0.0 } else { (-y / h) / h }
    }
);

// --- logaddexp -------------------------------------------------------------

float_pair!(
    /// `d/da log(e^a + e^b) = sigmoid(a - b)`, taken from the operands rather
    /// than as `exp(a - out)`, which is `exp(inf - inf)` wherever `a` is the
    /// infinity the output took. The limits are kept at the infinities: an
    /// operand at `+inf` takes the whole slope (half of it each when both
    /// are), and two operands at `-inf` take none, as an all `-inf` row of
    /// `logsumexp` does.
    pub(crate) LOGADDEXP_D_LHS, |a, b| {
        if a == b {
            if a.is_infinite() && a < 0.0 { 0.0 } else { 0.5 }
        } else {
            let d = a - b;
            if d >= 0.0 {
                1.0 / (1.0 + (-d).exp())
            } else {
                let e = d.exp();
                e / (1.0 + e)
            }
        }
    }
);
float_pair!(
    /// `d/db log(e^a + e^b)`, which is [`LOGADDEXP_D_LHS`] with the operands
    /// swapped.
    pub(crate) LOGADDEXP_D_RHS, |a, b| {
        if a == b {
            if b.is_infinite() && b < 0.0 { 0.0 } else { 0.5 }
        } else {
            let d = b - a;
            if d >= 0.0 {
                1.0 / (1.0 + (-d).exp())
            } else {
                let e = d.exp();
                e / (1.0 + e)
            }
        }
    }
);

// --- logaddexp2 ------------------------------------------------------------

/// `log2(2^a + 2^b)` in float64: the larger operand plus the base-2 `log1p` of
/// the smaller's power relative to it, so neither power is formed.
///
/// The exponent difference goes to `exp2` as it is. Scaling the operands by
/// `ln 2` first and reusing the natural form rounds an error of up to
/// `|a| * eps` into the exponent, which the power then carries into the answer
/// as a relative error: at `(0, -1000)` that was 445 ulp.
///
/// Equal operands are `a + 1` exactly, which is also what keeps two equal
/// infinities from differing by NaN. Anything else at an infinity falls out of
/// the arithmetic: a `-inf` operand's power is zero, and a `+inf` one is the
/// larger operand.
fn logaddexp2_f64(a: f64, b: f64) -> f64 {
    if a == b {
        return a + 1.0;
    }
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    high + (low - high).exp2().ln_1p() * std::f64::consts::LOG2_E
}

/// `d/da log2(2^a + 2^b) = 2^a / (2^a + 2^b)`, the logistic function of
/// `(a - b) ln 2`, with [`LOGADDEXP_D_LHS`]'s limits at the infinities. Formed
/// in float64 at either width, so a float32 difference does not round its
/// error into the power the way the forward's used to.
fn logaddexp2_d_lhs(a: f64, b: f64) -> f64 {
    if a == b {
        return if a.is_infinite() && a < 0.0 { 0.0 } else { 0.5 };
    }
    let d = a - b;
    if d >= 0.0 {
        1.0 / (1.0 + (-d).exp2())
    } else {
        let e = d.exp2();
        e / (1.0 + e)
    }
}

/// See [`logaddexp2_d_lhs`].
const LOGADDEXP2_D_LHS: FloatBinaryKernel = (
    |a, b| logaddexp2_d_lhs(a as f64, b as f64) as f32,
    logaddexp2_d_lhs,
);
/// `d/db`, which is [`LOGADDEXP2_D_LHS`] with the operands swapped.
const LOGADDEXP2_D_RHS: FloatBinaryKernel = (
    |a, b| logaddexp2_d_lhs(b as f64, a as f64) as f32,
    |a, b| logaddexp2_d_lhs(b, a),
);

// --- fmax and fmin ---------------------------------------------------------

float_kernel!(
    /// The larger of two values, ignoring a NaN in either.
    ///
    /// `maximum` propagates NaN, which is right for a comparison and wrong for
    /// a running extremum over data with holes in it. This was five passes of
    /// Python -- two `isnan`, a `maximum` and two `where` -- with four
    /// full-size temporaries between them, and the `where`s were the expensive
    /// part: a mask the branch predictor cannot guess costs six times one it
    /// can.
    ///
    /// The last arm is the tie. `-0.0` and `+0.0` compare equal, so either is
    /// "the larger", and a kernel that leaves the choice to whichever of its
    /// vectorized body or scalar tail met the pair gives the same two operands
    /// different signs at different array lengths.
    ///
    /// So the rule is IEEE 754-2019's `maximumNumber`: `-0.0` is below
    /// `+0.0`, which is the order every other part of this crate already sorts
    /// them in, and the answer does not depend on where in a buffer the
    /// operands sat.
    fmax_f32, fmax_f64, |x, y| {
        // `y >= x` is false when either is NaN, so the `x.is_nan()` test is
        // what picks `y` when only `x` is -- and when both are, which gives
        // NaN either way. Two comparisons and a select, no branch.
        let extremum = if y >= x || x.is_nan() { y } else { x };
        // The tie, settled without a branch either. The only distinguishable
        // values that compare equal are the two zeros, and `-0.0 + +0.0` is
        // `+0.0` while `-0.0 + -0.0` is `-0.0` -- which is the rule, spelled
        // as arithmetic. A chain of comparisons instead cost four times as
        // much: 0.47ms over a million float32 against 2.01.
        if x == 0.0 && y == 0.0 { x + y } else { extremum }
    }
);
float_kernel!(
    /// The smaller of two values, ignoring a NaN in either. See [`FMAX`], and
    /// note that the tie goes the other way: `fmin(-0.0, +0.0)` is `-0.0`.
    fmin_f32, fmin_f64, |x, y| {
        let extremum = if y <= x || x.is_nan() { y } else { x };
        // Mirrors `FMAX`, negated: `fmin` wants `-0.0` wherever `fmax` wants
        // `+0.0`, and negating both operands turns one rule into the other.
        if x == 0.0 && y == 0.0 { -((-x) + (-y)) } else { extremum }
    }
);
float_pair!(
    /// `d/dx fmax(x, y)`: one where `x` is what came out, zero where `y` was.
    ///
    /// The derivative ties to `x` where the value ties on the sign bit. That
    /// is not an inconsistency -- a tie means the two values are equal, so
    /// which one is "returned" is unobservable, while the derivative has to
    /// pick a side and the first operand is the conventional one. It is also
    /// the side the five passes this replaces picked, so no gradient moves.
    FMAX_D_X, |x, y| {
        if x.is_nan() {
            0.0
        } else if y.is_nan() || x >= y {
            1.0
        } else {
            0.0
        }
    }
);
float_pair!(
    /// `d/dy fmax(x, y)`, the complement of [`FMAX_D_X`].
    FMAX_D_Y, |x, y| {
        if x.is_nan() {
            1.0
        } else if y.is_nan() || x >= y {
            0.0
        } else {
            1.0
        }
    }
);
float_pair!(
    /// `d/dx fmin(x, y)`. See [`FMAX_D_X`].
    FMIN_D_X, |x, y| {
        if x.is_nan() {
            0.0
        } else if y.is_nan() || x <= y {
            1.0
        } else {
            0.0
        }
    }
);
float_pair!(
    /// `d/dy fmin(x, y)`, the complement of [`FMIN_D_X`].
    FMIN_D_Y, |x, y| {
        if x.is_nan() {
            1.0
        } else if y.is_nan() || x <= y {
            0.0
        } else {
            1.0
        }
    }
);

// --- hypot -----------------------------------------------------------------

float_kernel!(
    /// `sqrt(x^2 + y^2)` without forming either square, so it answers for
    /// operands whose squares would overflow or flush to zero.
    hypot_f32, hypot_f64, |x, y| x.hypot(y)
);
float_pair!(
    /// `d/dx hypot(x, y) = x / hypot(x, y)`. The origin is a cone point with
    /// no derivative; it takes the zero subgradient, as `norm` does at the
    /// same point, rather than the `0 / 0` the formula reads there.
    HYPOT_D_X, |x, y| {
        let h = x.hypot(y);
        if h == 0.0 { 0.0 } else { x / h }
    }
);
float_pair!(
    /// `d/dy hypot(x, y) = y / hypot(x, y)`, 0 at the origin as in `HYPOT_D_X`.
    HYPOT_D_Y, |x, y| {
        let h = x.hypot(y);
        if h == 0.0 { 0.0 } else { y / h }
    }
);

// --- copysign --------------------------------------------------------------

float_kernel!(
    /// The magnitude of `x` with the sign of `y`, signed zeros and NaN
    /// included -- `copysign(1, -0.0)` is `-1`.
    copysign_f32, copysign_f64, |x, y| x.copysign(y)
);
float_pair!(
    /// `d/dx copysign(x, y)` is `+1` where the sign is kept and `-1` where it
    /// is flipped. `signum` reads the sign bit, so `-0.0` counts as negative.
    COPYSIGN_D_X, |x, y| x.signum() * y.signum()
);
float_pair!(
    /// `copysign` does not vary with the *magnitude* of `y`, only with its
    /// sign bit, which no derivative can see.
    COPYSIGN_D_Y, |_x, _y| 0.0
);

// --- xlogy -----------------------------------------------------------------

float_kernel!(
    /// `x * log(y)`, defined as `0` wherever `x` is zero.
    ///
    /// That is the limit the entropy and cross-entropy formulas need: the
    /// plain product gives `0 * -inf = NaN` at `x = 0, y = 0`, which is the
    /// case those formulas hit most. A NaN `y` still poisons the result, since
    /// then there is no limit to take.
    xlogy_f32, xlogy_f64, |x, y| {
        if x == 0.0 && !y.is_nan() {
            0.0
        } else {
            x * y.ln()
        }
    }
);
float_pair!(
    /// `d/dx (x log y) = log y`.
    XLOGY_D_X, |_x, y| y.ln()
);
float_pair!(
    /// `d/dy (x log y) = x / y`, and 0 wherever `x` is zero: the value is the
    /// constant 0 along `y` there, though `x / y` reads `0 / 0` at `y = 0`.
    XLOGY_D_Y, |x, y| {
        if x == 0.0 && !y.is_nan() { 0.0 } else { x / y }
    }
);

// --- heaviside -------------------------------------------------------------

float_kernel!(
    /// The step: `0` below zero, `1` above it, and `other` exactly at it --
    /// which is the whole reason it takes a second operand, since that value
    /// is the one convention never agrees on.
    ///
    /// A NaN input stays NaN; it is on neither side of the step.
    heaviside_f32, heaviside_f64, |x, at_zero| {
        if x < 0.0 {
            0.0
        } else if x > 0.0 {
            1.0
        } else if x == 0.0 {
            at_zero
        } else {
            x
        }
    }
);
float_pair!(
    /// The step is flat wherever it is defined, so the gradient with respect
    /// to the input is zero everywhere the derivative exists -- and the
    /// jump at zero has no finite derivative to report.
    HEAVISIDE_D_X, |_x, _at_zero| 0.0
);
float_pair!(
    /// The second operand is the value taken at exactly zero, so it reaches
    /// the output there and nowhere else.
    HEAVISIDE_D_AT_ZERO, |x, _at_zero| if x == 0.0 { 1.0 } else { 0.0 }
);

// --- nextafter -------------------------------------------------------------

/// Defines `nextafter` at one width. The bit manipulation names the type on
/// every line, so this cannot go through [`float_pair!`], which requires a
/// body that typechecks at both.
macro_rules! next_after_fn {
    ($name:ident, $ty:ty) => {
        #[inline(always)]
        fn $name(from: $ty, towards: $ty) -> $ty {
            if from.is_nan() || towards.is_nan() {
                return <$ty>::NAN;
            }
            if from == towards {
                // Includes `0.0` against `-0.0`, where the answer is the
                // destination rather than a step in either direction.
                return towards;
            }
            if from == 0.0 {
                // Neither zero has a neighbour one bit away in the direction
                // asked for; the smallest subnormal does.
                let smallest = <$ty>::from_bits(1);
                return if towards > 0.0 { smallest } else { -smallest };
            }
            // Consecutive representable values are consecutive bit patterns
            // within a sign, and the pattern grows away from zero. So a step
            // towards a larger value adds one bit above zero and subtracts one
            // below it.
            let bits = from.to_bits();
            let stepped = if (from < towards) == (from > 0.0) {
                bits + 1
            } else {
                bits - 1
            };
            <$ty>::from_bits(stepped)
        }
    };
}

next_after_fn!(next_after_f32, f32);
next_after_fn!(next_after_f64, f64);

float_pair!(
    /// `nextafter(x, y)` differs from `x` by a single ulp, so as a
    /// real-valued function it is the identity and this is its slope. The
    /// step itself is below anything a derivative can see.
    NEXTAFTER_D_FROM, |_x, _y| 1.0
);
float_pair!(
    /// Only the *direction* of the second operand reaches the answer, and no
    /// derivative can see a direction.
    NEXTAFTER_D_TOWARDS, |_x, _y| 0.0
);

// --- ldexp -----------------------------------------------------------------

/// `x * 2^n` by exponent arithmetic: one rounding, and only where the answer
/// is subnormal, however far the exponent reaches. Past +-400 every float32
/// answer is already zero or infinite, so the exponent is clamped there to fit
/// the `i32` the scaling takes.
#[inline(always)]
fn ldexp_f32(x: f32, n: f32) -> f32 {
    libm::scalbnf(x, n.clamp(-400.0, 400.0) as i32)
}

/// See [`ldexp_f32`]; float64 saturates past +-2200.
#[inline(always)]
fn ldexp_f64(x: f64, n: f64) -> f64 {
    libm::scalbn(x, n.clamp(-2200.0, 2200.0) as i32)
}

/// `d/dx x * 2^n = 2^n`, exact for the same reason the forward is.
const LDEXP_D_X: FloatBinaryKernel = (|_x, n| ldexp_f32(1.0, n), |_x, n| ldexp_f64(1.0, n));
/// `d/dn x * 2^n = x * 2^n * ln 2`. Only an integer exponent reaches the
/// kernel, and an integer tensor never asks for a gradient; this is the slope
/// of the same function read continuously, so the pair stays complete.
const LDEXP_D_N: FloatBinaryKernel = (
    |x, n| ldexp_f32(x, n) * std::f32::consts::LN_2,
    |x, n| ldexp_f64(x, n) * std::f64::consts::LN_2,
);

/// `input * 2^other`.
///
/// An integer `other` is applied to the exponent directly. It was the product
/// `input * exp2(other)`, which forms the power on its own first: `2^-1100`
/// is zero and `2^1100` infinite in float64, so `ldexp(2^1000, -1100)` came
/// back 0 and `ldexp(2^-1000, 1100)` infinite where both answers are ordinary
/// numbers, `ldexp(3, -1075)` was 0 rather than the smallest subnormal but
/// one, and `ldexp(0, 5000)` was `0 * inf`, NaN. A float `input` keeps its
/// dtype, where the product had widened float32 to float64; an integer one
/// is read as float64, which holds every integer it could have been scaled
/// from exactly.
///
/// A float `other` may be fractional and is differentiable, which makes it a
/// power rather than an exponent: it is still the product, in float64.
pub fn ldexp(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if rhs.dtype().is_float() {
        let power = crate::ops::special::exp2(&rhs.astype(DataType::Float64)?)?;
        return crate::ops::arithmetic::mul(lhs, &power);
    }
    if lhs.dtype().is_float() {
        float_binary_mono(lhs, rhs, ldexp_f32, ldexp_f64, [LDEXP_D_X, LDEXP_D_N])
    } else {
        let widened = lhs.astype(DataType::Float64)?;
        float_binary_mono(&widened, rhs, ldexp_f32, ldexp_f64, [LDEXP_D_X, LDEXP_D_N])
    }
}

/// Runs one element-wise binary float function and records its chain rule.
///
/// The forward arrives as two concrete function items rather than as a
/// [`FloatBinaryKernel`], so the per-element call inlines into the broadcast
/// loop instead of going through a pointer -- see that type for the
/// measurement. The partials stay pointers: the backward is not hot, and
/// monomorphizing it would mean a `FloatBinaryBackward` per op.
fn float_binary_mono<N, W>(
    lhs: &Tensor,
    rhs: &Tensor,
    narrow: N,
    wide: W,
    partials: [FloatBinaryKernel; 2],
) -> Result<Tensor>
where
    N: Fn(f32, f32) -> f32 + Send + Sync,
    W: Fn(f64, f64) -> f64 + Send + Sync,
{
    float_binary_with(lhs, rhs, partials, |l, r, dtype, shape| {
        float_binary_data_with(l, r, dtype, shape, narrow, wide)
    })
}

/// [`float_binary_mono`] for a kernel that is mostly selects, whose operands,
/// when neither needs broadcasting, go through
/// [`crate::ops::map::binary_map_selecting`] and so get the AVX-512 build.
/// `fmin` and `fmax` are three compares and three selects a lane.
fn float_binary_selecting<N, W>(
    lhs: &Tensor,
    rhs: &Tensor,
    narrow: N,
    wide: W,
    partials: [FloatBinaryKernel; 2],
) -> Result<Tensor>
where
    N: Fn(f32, f32) -> f32 + Send + Sync,
    W: Fn(f64, f64) -> f64 + Send + Sync,
{
    use crate::ops::map::binary_map_selecting;
    float_binary_with(lhs, rhs, partials, |l, r, dtype, shape| {
        if l.shape().dims() == shape.dims() && r.shape().dims() == shape.dims() {
            let device = l.device();
            if let (Some(a), Some(b)) = (l.data().as_f32_slice(), r.data().as_f32_slice()) {
                let out = binary_map_selecting(a, b, narrow);
                return Ok(TensorData::from_vec::<f32>(out, DataType::Float32, device));
            }
            if let (Some(a), Some(b)) = (l.data().as_f64_slice(), r.data().as_f64_slice()) {
                let out = binary_map_selecting(a, b, wide);
                return Ok(TensorData::from_vec::<f64>(out, DataType::Float64, device));
            }
        }
        float_binary_data_with(l, r, dtype, shape, narrow, wide)
    })
}

/// The part every float binary op shares: promote and broadcast the operands,
/// compute the forward with `data`, and record the backward from `partials`.
fn float_binary_with<D>(
    lhs: &Tensor,
    rhs: &Tensor,
    partials: [FloatBinaryKernel; 2],
    data: D,
) -> Result<Tensor>
where
    D: FnOnce(&Tensor, &Tensor, DataType, &Shape) -> Result<TensorData>,
{
    let (lhs_cast, rhs_cast, dtype, output_shape) =
        coerce_and_broadcast(lhs, rhs, BinaryOpKind::Div)?;
    let lhs_tensor = lhs_cast.into_owned();
    let rhs_tensor = rhs_cast.into_owned();

    let output_data = data(&lhs_tensor, &rhs_tensor, dtype, &output_shape)?;
    attach_float_binary_grad(
        lhs,
        rhs,
        lhs_tensor,
        rhs_tensor,
        dtype,
        output_shape,
        output_data,
        partials,
    )
}

/// [`float_binary_mono`] with a float32 kernel that wants whole blocks.
///
/// Some forwards are not made faster by inlining because their body is a
/// `libm` call: `atan2` costs 5.5ms over a million float32 where the `atan`
/// kernel beside it costs 0.32, and no amount of monomorphizing a call to
/// `atan2f` changes that. Those need the vectorized kernel, which works on
/// slices rather than elements, so this takes one -- used whenever the operands
/// are float32, broadcast or not (see
/// [`crate::ops::kernels::broadcast_binary_blocks`]). Float64 is offered whole
/// to the installed provider as `wide_op` (see `ops::provider`), if the op
/// names one, when neither operand needs repeating beyond a single element,
/// and otherwise runs `wide` element-wise.
///
/// # Safety
///
/// `blocks` must initialize every element of the output block it is handed;
/// a call to one [`crate::ops::simd::F32Kernel`] method does.
unsafe fn float_binary_blocked<W, B>(
    lhs: &Tensor,
    rhs: &Tensor,
    wide: W,
    wide_op: Option<crate::ops::provider::Ufunc>,
    blocks: B,
    partials: [FloatBinaryKernel; 2],
) -> Result<Tensor>
where
    W: Fn(f64, f64) -> f64 + Send + Sync,
    B: Fn(&[f32], &[f32], &mut [std::mem::MaybeUninit<f32>]) + Send + Sync,
{
    let (lhs_cast, rhs_cast, dtype, output_shape) =
        coerce_and_broadcast(lhs, rhs, BinaryOpKind::Div)?;
    let lhs_tensor = lhs_cast.into_owned();
    let rhs_tensor = rhs_cast.into_owned();

    // A broadcast takes the kernel too, each task gathering its stretch of
    // the operands into runs that match its block. That is a copy at a
    // fraction of a nanosecond an element against the several a scalar
    // `libm` call costs: `atan2` of a million float32 against one row went
    // from 7.8ms to 0.63.
    let output_data = if dtype == DataType::Float32 {
        let (Some(left), Some(right)) = (
            lhs_tensor.data().as_f32_slice(),
            rhs_tensor.data().as_f32_slice(),
        ) else {
            return Err(MinitensorError::internal_error(
                "float32 operands without float32 storage",
            ));
        };
        // SAFETY: `blocks` writes every element of each block it is given,
        // by this function's contract.
        let out = unsafe {
            crate::ops::kernels::broadcast_binary_blocks(
                left,
                right,
                lhs_tensor.shape(),
                rhs_tensor.shape(),
                &output_shape,
                crate::ops::map::VECTOR_F32_PAR_THRESHOLD,
                blocks,
            )
        };
        TensorData::from_vec::<f32>(out, DataType::Float32, lhs.device())
    } else {
        // Promotion by `/` leaves float64 as the only other dtype.
        match wide_op.and_then(|op| offer_wide(&lhs_tensor, &rhs_tensor, &output_shape, op)) {
            Some(out) => TensorData::from_vec::<f64>(out, DataType::Float64, lhs.device()),
            None => broadcast_binary_arm!(
                &lhs_tensor,
                &rhs_tensor,
                &output_shape,
                as_f64_slice,
                "f64",
                wide
            ),
        }
    };
    attach_float_binary_grad(
        lhs,
        rhs,
        lhs_tensor,
        rhs_tensor,
        dtype,
        output_shape,
        output_data,
        partials,
    )
}

/// A float64 pair's answer from the installed provider, or `None` to compute it
/// here. Offered only when each operand is the output's length -- its flat
/// order is then the output's, whatever leading ones its shape has -- or a
/// single element, which is the one broadcast a provider is asked to know.
fn offer_wide(
    lhs: &Tensor,
    rhs: &Tensor,
    output_shape: &Shape,
    op: crate::ops::provider::Ufunc,
) -> Option<Vec<f64>> {
    let len = output_shape.numel();
    let spans = |values: &[f64]| values.len() == len || values.len() == 1;
    let (left, right) = (lhs.data().as_f64_slice()?, rhs.data().as_f64_slice()?);
    if !(spans(left) && spans(right)) {
        return None;
    }
    crate::ops::provider::offer_binary_f64(op, left, right, len)
}

/// Wrap one of these kernels' output as a tensor and record its chain rule.
#[allow(clippy::too_many_arguments)]
fn attach_float_binary_grad(
    lhs: &Tensor,
    rhs: &Tensor,
    lhs_tensor: Tensor,
    rhs_tensor: Tensor,
    dtype: DataType,
    output_shape: Shape,
    output_data: TensorData,
    partials: [FloatBinaryKernel; 2],
) -> Result<Tensor> {
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        dtype,
        lhs.device(),
        requires_grad,
    );

    if requires_grad {
        return with_grad_fn(
            output,
            Arc::new(FloatBinaryBackward {
                lhs: lhs_tensor.detach(),
                rhs: rhs_tensor.detach(),
                input_ids: [lhs.id(), rhs.id()],
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
                partials,
            }),
        );
    }

    Ok(output)
}

/// Applies `kernel` element-wise over two operands already promoted to
/// `dtype`, with broadcasting.
pub(crate) fn float_binary_data(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DataType,
    output_shape: &Shape,
    kernel: FloatBinaryKernel,
) -> Result<TensorData> {
    float_binary_data_with(lhs, rhs, dtype, output_shape, kernel.0, kernel.1)
}

/// [`float_binary_data`] taking its two halves as generic parameters rather
/// than as function pointers.
///
/// Handed concrete function *items* -- `fmax_of::<f32>` rather than a
/// `FloatBinaryKernel` field -- this instantiates per kernel and the
/// per-element call inlines into the loop. Handed the fields, it is the
/// pointer version and nothing is lost; that is what the wrapper above does.
pub(crate) fn float_binary_data_with<N, W>(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DataType,
    output_shape: &Shape,
    narrow: N,
    wide: W,
) -> Result<TensorData>
where
    N: Fn(f32, f32) -> f32 + Send + Sync,
    W: Fn(f64, f64) -> f64 + Send + Sync,
{
    Ok(match dtype {
        DataType::Float32 => {
            broadcast_binary_arm!(lhs, rhs, output_shape, as_f32_slice, "f32", narrow)
        }
        DataType::Float64 => {
            broadcast_binary_arm!(lhs, rhs, output_shape, as_f64_slice, "f64", wide)
        }
        other => {
            return Err(MinitensorError::internal_error(format!(
                "float binary kernel reached with dtype {other}, which promotion cannot produce"
            )));
        }
    })
}

/// [`float_binary_data`] wrapped as a tensor, for the backward pass to
/// evaluate one partial derivative in a single walk over the operands.
pub(crate) fn float_binary_tensor(
    lhs: &Tensor,
    rhs: &Tensor,
    kernel: FloatBinaryKernel,
) -> Result<Tensor> {
    let output_shape = lhs.shape().broadcast_with_dtype(rhs.shape(), lhs.dtype())?;
    let data = float_binary_data(lhs, rhs, lhs.dtype(), &output_shape, kernel)?;
    Ok(Tensor::new(
        Arc::new(data),
        output_shape,
        lhs.dtype(),
        lhs.device(),
        false,
    ))
}

/// Declares one public op from its forward kernel and its two partials.
macro_rules! float_binary_op {
    ($name:ident, $narrow:ident, $wide:ident, $d_lhs:ident, $d_rhs:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            float_binary_mono(lhs, rhs, $narrow, $wide, [$d_lhs, $d_rhs])
        }
    };
}

/// The angle of the point `(other, input)` from the positive x-axis, in
/// `(-pi, pi]`, keeping the quadrant that `atan(input / other)` loses.
///
/// Float32 goes through `ops::simd::transcendental`, which computes it as
/// `atan(y / x)` in float64 and narrows once. `atan2f` took 5.5ms over a
/// million elements where the `atan` kernel it is one division away from costs
/// 0.32.
pub fn atan2(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: the block closure is one `F32Kernel` method call.
    unsafe {
        float_binary_blocked(
            lhs,
            rhs,
            f64::atan2,
            Some(crate::ops::provider::Ufunc::Atan2),
            move |y, x, out| kernel.atan2(y, x, out),
            [ATAN2_D_Y, ATAN2_D_X],
        )
    }
}

/// `log2(2**lhs + 2**rhs)`, without forming either power.
///
/// Float32 goes through `ops::simd::transcendental`, which computes it in
/// float64 and rounds once; float64 is [`logaddexp2_f64`].
pub fn logaddexp2(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let kernel = crate::ops::simd::F32Kernel::select();
    // SAFETY: the block closure is one `F32Kernel` method call.
    unsafe {
        float_binary_blocked(
            lhs,
            rhs,
            logaddexp2_f64,
            None,
            move |a, b, out| kernel.logaddexp2(a, b, out),
            [LOGADDEXP2_D_LHS, LOGADDEXP2_D_RHS],
        )
    }
}

/// Element-wise larger of two tensors, ignoring a NaN in either operand. NaN
/// only where both are.
///
/// An integer or boolean pair has no NaN to skip, so it is a plain `maximum` --
/// which also keeps its dtype, where this family's promotion would turn two
/// `int64` operands into a float the way `/` does.
pub fn fmax(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if !lhs.dtype().is_float() && !rhs.dtype().is_float() {
        return crate::ops::minmax::maximum(lhs, rhs);
    }
    float_binary_selecting(lhs, rhs, fmax_f32, fmax_f64, [FMAX_D_X, FMAX_D_Y])
}

/// Element-wise smaller of two tensors, ignoring a NaN in either operand. NaN
/// only where both are. See [`fmax`].
pub fn fmin(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if !lhs.dtype().is_float() && !rhs.dtype().is_float() {
        return crate::ops::minmax::minimum(lhs, rhs);
    }
    float_binary_selecting(lhs, rhs, fmin_f32, fmin_f64, [FMIN_D_X, FMIN_D_Y])
}

float_binary_op!(
    hypot,
    hypot_f32,
    hypot_f64,
    HYPOT_D_X,
    HYPOT_D_Y,
    "`sqrt(input^2 + other^2)`, computed without forming either square, so it \
     answers where the squares would overflow."
);
float_binary_op!(
    copysign,
    copysign_f32,
    copysign_f64,
    COPYSIGN_D_X,
    COPYSIGN_D_Y,
    "The magnitude of `input` with the sign of `other`, signed zeros included."
);
float_binary_op!(
    xlogy,
    xlogy_f32,
    xlogy_f64,
    XLOGY_D_X,
    XLOGY_D_Y,
    "`input * log(other)`, taken as `0` wherever `input` is zero rather than \
     as the `0 * -inf` the plain product would give."
);
float_binary_op!(
    heaviside,
    heaviside_f32,
    heaviside_f64,
    HEAVISIDE_D_X,
    HEAVISIDE_D_AT_ZERO,
    "The unit step of `input`, taking the value `other` at exactly zero."
);
float_binary_op!(
    nextafter,
    next_after_f32,
    next_after_f64,
    NEXTAFTER_D_FROM,
    NEXTAFTER_D_TOWARDS,
    "The next representable value after `input` in the direction of `other`."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::vector;
    use crate::{autograd::backward_collect, device::Device, tensor::TensorData};

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        vector(data)
    }

    fn i64_tensor(data: Vec<i64>) -> Tensor {
        vector(data)
    }

    fn wide(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    /// Central differences of `op` with respect to one operand.
    fn numeric_grad(
        op: fn(&Tensor, &Tensor) -> Result<Tensor>,
        lhs: &[f64],
        rhs: &[f64],
        which: usize,
    ) -> Vec<f64> {
        let eps = 1e-6;
        let mut grads = vec![0.0; lhs.len()];
        for i in 0..lhs.len() {
            let mut plus = [lhs.to_vec(), rhs.to_vec()];
            let mut minus = [lhs.to_vec(), rhs.to_vec()];
            plus[which][i] += eps;
            minus[which][i] -= eps;
            let up = wide(&op(&f64_tensor(plus[0].clone()), &f64_tensor(plus[1].clone())).unwrap());
            let down =
                wide(&op(&f64_tensor(minus[0].clone()), &f64_tensor(minus[1].clone())).unwrap());
            grads[i] = (up[i] - down[i]) / (2.0 * eps);
        }
        grads
    }

    fn analytic_grads(
        op: fn(&Tensor, &Tensor) -> Result<Tensor>,
        lhs: &[f64],
        rhs: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let a = f64_tensor(lhs.to_vec()).requires_grad_(true);
        let b = f64_tensor(rhs.to_vec()).requires_grad_(true);
        let out = op(&a, &b).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        (
            wide(grads.get(&a.id()).unwrap()),
            wide(grads.get(&b.id()).unwrap()),
        )
    }

    #[test]
    fn atan2_covers_every_quadrant_and_the_axes() {
        let y = vec![1.0, 1.0, -1.0, -1.0, 0.0, -0.0, 1.0, -1.0];
        let x = vec![1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0];
        let got = wide(&atan2(&f64_tensor(y.clone()), &f64_tensor(x.clone())).unwrap());
        for (i, (&yi, &xi)) in y.iter().zip(&x).enumerate() {
            assert_eq!(got[i], yi.atan2(xi), "atan2({yi}, {xi})");
        }
        // The quadrant is the point of the op: atan(y/x) collapses the second
        // and fourth quadrants onto the first and third.
        assert!(got[1] > 0.0 && got[2] < 0.0);
    }

    #[test]
    fn hypot_survives_operands_whose_squares_overflow() {
        let big = 1e200_f64;
        let got = wide(&hypot(&f64_tensor(vec![big, 3.0]), &f64_tensor(vec![big, 4.0])).unwrap());
        assert!(got[0].is_finite(), "hypot overflowed: {}", got[0]);
        assert!((got[0] - big * std::f64::consts::SQRT_2).abs() < big * 1e-15);
        assert_eq!(got[1], 5.0);

        // ...and operands whose squares would flush to zero.
        let tiny = 1e-200_f64;
        let got = wide(&hypot(&f64_tensor(vec![tiny]), &f64_tensor(vec![tiny])).unwrap());
        assert!(got[0] > 0.0, "hypot underflowed to {}", got[0]);
    }

    #[test]
    fn copysign_reads_the_sign_bit_of_zero() {
        let x = vec![3.0, 3.0, -3.0, -3.0, 0.0];
        let y = vec![1.0, -0.0, 0.0, -1.0, -1.0];
        let got = wide(&copysign(&f64_tensor(x.clone()), &f64_tensor(y.clone())).unwrap());
        assert_eq!(got[..4], [3.0, -3.0, 3.0, -3.0]);
        assert!(got[4].is_sign_negative() && got[4] == 0.0);
    }

    #[test]
    fn xlogy_is_zero_where_the_plain_product_is_nan() {
        let x = vec![0.0, 0.0, 0.0, 2.0, 2.0];
        let y = vec![0.0, f64::INFINITY, 1.0, std::f64::consts::E, 0.0];
        let got = wide(&xlogy(&f64_tensor(x), &f64_tensor(y)).unwrap());
        assert_eq!(got[0], 0.0, "0 * log(0) must be the limit, not NaN");
        assert_eq!(got[1], 0.0);
        assert_eq!(got[2], 0.0);
        assert!((got[3] - 2.0).abs() < 1e-15);
        assert_eq!(got[4], f64::NEG_INFINITY);

        // A NaN second operand has no limit to take, so it survives.
        let got = wide(&xlogy(&f64_tensor(vec![0.0]), &f64_tensor(vec![f64::NAN])).unwrap());
        assert!(got[0].is_nan());
    }

    #[test]
    fn gradients_match_central_differences() {
        // Each op is sampled away from its own singular set: no origin for
        // `atan2` and `hypot`, and a strictly positive second operand for
        // `xlogy`, whose derivative is the logarithm of it.
        let lhs = [0.7, -1.3, 2.4, -0.4];
        let signed = [1.1, 0.6, -2.2, 3.5];
        let positive = [1.1, 0.6, 2.2, 3.5];

        for (name, op, rhs) in [
            (
                "atan2",
                atan2 as fn(&Tensor, &Tensor) -> Result<Tensor>,
                signed,
            ),
            ("hypot", hypot, signed),
            ("xlogy", xlogy, positive),
        ] {
            let (d_lhs, d_rhs) = analytic_grads(op, &lhs, &rhs);
            for (which, analytic) in [(0, d_lhs), (1, d_rhs)] {
                let numeric = numeric_grad(op, &lhs, &rhs, which);
                for (i, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
                    assert!(
                        (a - n).abs() <= 1e-5 * (1.0 + n.abs()),
                        "{name} d/d{which} [{i}]: analytic {a}, numeric {n}"
                    );
                }
            }
        }

        // `copysign` moves with the *sign* of its second operand, which no
        // central difference can see, so its derivative there is zero
        // everywhere it exists and only the first operand's is checkable.
        let (d_x, d_y) = analytic_grads(copysign, &lhs, &signed);
        assert_eq!(d_y, vec![0.0; 4]);
        for (i, &g) in d_x.iter().enumerate() {
            let expected = lhs[i].signum() * signed[i].signum();
            assert_eq!(g, expected, "copysign d/dx [{i}]");
        }
    }

    #[test]
    fn integer_operands_promote_to_float_and_broadcast() {
        let y = i64_tensor(vec![3, 4]);
        let x = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![4], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        let got = hypot(&y, &x).unwrap();
        // `/` widens two integers by width -- `int64` to `float64` -- and so do
        // these.
        assert_eq!(got.dtype(), DataType::Float64);
        assert_eq!(got.data().as_f64_slice().unwrap(), &[5.0, 32f64.sqrt()]);

        // A float64 operand pulls the result up, as it does for `/`.
        let got = hypot(&y, &f64_tensor(vec![4.0])).unwrap();
        assert_eq!(got.dtype(), DataType::Float64);
        assert_eq!(got.shape().dims(), &[2]);
    }

    #[test]
    fn a_frozen_operand_gets_no_gradient() {
        let a = f64_tensor(vec![1.0, 2.0]).requires_grad_(true);
        let b = f64_tensor(vec![3.0, 4.0]);
        let out = atan2(&a, &b).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        assert!(grads.contains_key(&a.id()));
        assert!(!grads.contains_key(&b.id()));
    }

    #[test]
    fn empty_and_mismatched_shapes() {
        let empty = f64_tensor(vec![]);
        assert_eq!(hypot(&empty, &empty).unwrap().shape().dims(), &[0]);

        let a = f64_tensor(vec![1.0, 2.0, 3.0]);
        let b = f64_tensor(vec![1.0, 2.0]);
        assert!(atan2(&a, &b).is_err());
    }

    #[test]
    fn heaviside_steps_at_zero_and_takes_its_second_operand_there() {
        let x = f64_tensor(vec![-2.0, -0.0, 0.0, 3.5, f64::NAN]);
        let at_zero = f64_tensor(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let got = wide(&heaviside(&x, &at_zero).unwrap());
        assert_eq!(got[0], 0.0);
        // Negative zero is still zero, so it takes the given value too.
        assert_eq!(got[1], 0.5);
        assert_eq!(got[2], 0.5);
        assert_eq!(got[3], 1.0);
        assert!(got[4].is_nan(), "a NaN is on neither side of the step");
    }

    #[test]
    fn the_heaviside_gradient_reaches_only_the_value_at_zero() {
        let x = f64_tensor(vec![-1.0, 0.0, 1.0]).requires_grad_(true);
        let at_zero = f64_tensor(vec![0.25, 0.25, 0.25]).requires_grad_(true);
        let out = heaviside(&x, &at_zero).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        // Flat wherever it is defined.
        assert_eq!(wide(grads.get(&x.id()).unwrap()), vec![0.0; 3]);
        // And the second operand is the output at exactly zero, nowhere else.
        assert_eq!(wide(grads.get(&at_zero.id()).unwrap()), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn nextafter_moves_exactly_one_representable_value() {
        let from = f64_tensor(vec![1.0, 1.0, -1.0, -1.0]);
        let towards = f64_tensor(vec![2.0, 0.0, 0.0, -2.0]);
        let got = wide(&nextafter(&from, &towards).unwrap());

        // One ulp up and down from 1, and the same either side of -1.
        assert_eq!(got[0], f64::from_bits(1.0f64.to_bits() + 1));
        assert_eq!(got[1], f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(got[2], -f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(got[3], -f64::from_bits(1.0f64.to_bits() + 1));

        // Every step is strictly in the direction asked for, and nothing sits
        // between the answer and where it started.
        assert!(got[0] > 1.0 && got[1] < 1.0);
        assert!(got[2] > -1.0 && got[3] < -1.0);
    }

    #[test]
    fn nextafter_handles_the_values_with_no_neighbour_of_their_own() {
        let from = f64_tensor(vec![0.0, -0.0, 0.0, f64::MAX, f64::INFINITY, 3.0, f64::NAN]);
        let towards = f64_tensor(vec![1.0, 1.0, -1.0, f64::INFINITY, 0.0, 3.0, 1.0]);
        let got = wide(&nextafter(&from, &towards).unwrap());

        // Neither zero has a neighbour a bit away; the smallest subnormal does.
        assert_eq!(got[0], f64::from_bits(1));
        assert_eq!(got[1], f64::from_bits(1));
        assert_eq!(got[2], -f64::from_bits(1));
        // Past the largest finite value there is only infinity, and back from
        // infinity there is only the largest finite value.
        assert_eq!(got[3], f64::INFINITY);
        assert_eq!(got[4], f64::MAX);
        // Already there, so it stays.
        assert_eq!(got[5], 3.0);
        assert!(got[6].is_nan());
    }

    #[test]
    fn nextafter_agrees_with_itself_at_float32() {
        let from = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, -1.0, 0.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let towards = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![2.0, -2.0, 1.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let got = nextafter(&from, &towards).unwrap();
        assert_eq!(got.dtype(), DataType::Float32);
        let values = got.data().as_f32_slice().unwrap();
        assert_eq!(values[0], f32::from_bits(1.0f32.to_bits() + 1));
        assert_eq!(values[1], -f32::from_bits(1.0f32.to_bits() + 1));
        assert_eq!(values[2], f32::from_bits(1));
    }

    #[test]
    fn the_nextafter_gradient_is_the_identity_in_its_first_operand() {
        let from = f64_tensor(vec![1.0, 2.0]).requires_grad_(true);
        let towards = f64_tensor(vec![5.0, 5.0]).requires_grad_(true);
        let out = nextafter(&from, &towards).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        // One ulp away from `from`, so as a real function this is the identity.
        assert_eq!(wide(grads.get(&from.id()).unwrap()), vec![1.0, 1.0]);
        // Only the direction of the second operand reaches the answer.
        assert_eq!(wide(grads.get(&towards.id()).unwrap()), vec![0.0, 0.0]);
    }
}
