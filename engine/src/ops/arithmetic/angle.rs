// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `deg2rad` and `rad2deg`, each element rounded once.
//!
//! Written as a product with `π / 180` or `180 / π`, the constant was
//! rounded -- to float32 for a float32 tensor -- and the product again, and
//! of random inputs about one in eight came out other than the float nearest
//! the exact angle, float32 and float64 alike. Each element is that float
//! now.
//!
//! A float32 answer is the float64 product with the constant's nearest
//! float64, within two float64 units of the exact one, and narrows to the
//! float32 nearest it wherever those units leave it clear of a float32
//! halfway point. A float64 answer is the product with the constant as a
//! double-double, the first product's error fused into the second, settled
//! by a bound on that rounding and on what the two halves leave of the
//! constant. Where neither settles it, the element's whole integer mantissa
//! is multiplied by the constant's leading 256 bits, and the bracket that
//! leaves is rounded at both ends. The exact product is irrational, and no
//! float64 or int64 brings it within 2^-141 of a halfway point -- scaled by
//! any power of two from 2^-16 to 2^16, either constant's continued fraction
//! has no partial quotient past 2^22 before its denominators pass 2^70 -- so
//! the ends, 2^-255 apart, round alike.

use std::mem::MaybeUninit;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::exact_sum::{Output, product_of, rounded_at};
use super::rounded_once::{Exact, product_error};
use crate::{
    autograd::{GradientFunction, TensorId, accumulate_grad, with_grad_fn},
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor, TensorData},
};

/// Degrees to radians: each element the float nearest `x · π / 180`. A
/// float32, int32 or bool tensor gives float32 and a float64 or int64 one
/// float64.
pub fn deg2rad(tensor: &Tensor) -> Result<Tensor> {
    scaled(tensor, &TO_RADIANS)
}

/// Radians to degrees: each element the float nearest `x · 180 / π`, the
/// dtypes as for [`deg2rad`].
pub fn rad2deg(tensor: &Tensor) -> Result<Tensor> {
    scaled(tensor, &TO_DEGREES)
}

/// One of the two constants: the float64 nearest it and the float64 nearest
/// what that leaves, a bound on what the two leave, and its leading 256 bits,
/// `bits · 2^-shift` truncated, little-endian with the top bit set.
struct Angle {
    high: f64,
    low: f64,
    left: f64,
    bits: [u64; 4],
    shift: i32,
}

/// `π / 180`.
const TO_RADIANS: Angle = Angle {
    high: f64::from_bits(0x3f91_df46_a252_9d39),
    low: f64::from_bits(0x3c15_c1d8_becd_d291),
    left: f64::from_bits(0x38b1_d937_fa42_8858),
    bits: [
        0xe502_a9b4_c94c_8512,
        0x00b7_aef5_01b5_e6b8,
        0x0ec5_f66e_9485_c4d9,
        0x8efa_3512_94e9_c8ae,
    ],
    shift: 261,
};

/// `180 / π`.
const TO_DEGREES: Angle = Angle {
    high: f64::from_bits(0x404c_a5dc_1a63_c1f8),
    low: f64::from_bits(0xbce1_e7ab_4564_05f9),
    left: f64::from_bits(0x398b_5051_96fa_bb42),
    bits: [
        0xcda2_7429_b138_0d91,
        0x3482_a25f_7cbf_02dc,
        0x0a97_537f_40d2_57d7,
        0xe52e_e0d3_1e0f_bdc3,
    ],
    shift: 250,
};

/// Elements decided at a time, a run with any left open taken again.
const RUN: usize = 256;

fn scaled(tensor: &Tensor, angle: &'static Angle) -> Result<Tensor> {
    let input = tensor.contiguous()?;
    let missing = || MinitensorError::internal_error("angle: a typed slice");
    let device = input.device();
    let (data, dtype) = match input.dtype() {
        DataType::Float32 => {
            let values = input.data().as_f32_slice().ok_or_else(missing)?;
            let out = runs(
                values,
                |x, out| narrow_run(x, out, angle),
                |x| narrow_value(f64::from(x), angle),
            );
            (
                TensorData::from_vec(out, DataType::Float32, device),
                DataType::Float32,
            )
        }
        DataType::Float64 => {
            let values = input.data().as_f64_slice().ok_or_else(missing)?;
            let out = runs(
                values,
                |x, out| wide_run(x, out, angle),
                |x| wide_value(x, angle),
            );
            (
                TensorData::from_vec(out, DataType::Float64, device),
                DataType::Float64,
            )
        }
        DataType::Int32 => {
            // Each exact in float64.
            let values = input.data().as_i32_slice().ok_or_else(missing)?;
            let out: Vec<f32> = values
                .iter()
                .map(|&x| narrow_value(f64::from(x), angle))
                .collect();
            (
                TensorData::from_vec(out, DataType::Float32, device),
                DataType::Float32,
            )
        }
        DataType::Int64 => {
            let values = input.data().as_i64_slice().ok_or_else(missing)?;
            let limit = 1i64 << 53;
            let out: Vec<f64> = values
                .iter()
                .map(|&x| {
                    if (-limit..=limit).contains(&x) {
                        wide_value(x as f64, angle)
                    } else {
                        exact_scaled(Exact::integer(x), angle)
                    }
                })
                .collect();
            (
                TensorData::from_vec(out, DataType::Float64, device),
                DataType::Float64,
            )
        }
        DataType::Bool => {
            let values = input.data().as_bool_slice().ok_or_else(missing)?;
            let one = narrow_value(1.0, angle);
            let out: Vec<f32> = values.iter().map(|&x| if x { one } else { 0.0 }).collect();
            (
                TensorData::from_vec(out, DataType::Float32, device),
                DataType::Float32,
            )
        }
    };
    let requires_grad =
        input.requires_grad() && input.dtype() == dtype && crate::autograd::is_grad_enabled();
    let output = Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        dtype,
        device,
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }
    let grad_fn = Arc::new(AngleBackward {
        angle,
        input_ids: [input.id()],
    });
    with_grad_fn(output, grad_fn)
}

/// `run` over the operand a run at a time, across the pool for a long one,
/// and a run it leaves anything open in again by `value`.
fn runs<I: Copy + Sync, T: Copy + Send + Sync>(
    values: &[I],
    run: impl Fn(&[I], &mut [MaybeUninit<T>]) -> bool + Sync,
    value: impl Fn(I) -> T + Sync,
) -> Vec<T> {
    let fill = |start: usize, block: &mut [MaybeUninit<T>]| {
        let input = &values[start..start + block.len()];
        for (input, out) in input.chunks(RUN).zip(block.chunks_mut(RUN)) {
            if run(input, out) {
                for (slot, &x) in out.iter_mut().zip(input) {
                    slot.write(value(x));
                }
            }
        }
    };
    // SAFETY: the chunks cover the output, and `fill` writes every element
    // of each.
    unsafe {
        crate::ops::map::build_vec(values.len(), |spare| {
            crate::ops::map::par_out_chunks(spare, 1 << 14, &fill)
        })
    }
}

/// The float32 nearest `x · C`: [`narrow_lane`], or where it leaves the
/// answer open, [`exact_float`].
fn narrow_value(x: f64, angle: &Angle) -> f32 {
    let (value, settled) = narrow_lane(x, angle);
    if settled {
        value
    } else {
        exact_float(x, angle)
    }
}

/// The float64 nearest `x · C`: [`wide_lane`], or where it leaves the answer
/// open, [`exact_float`].
fn wide_value(x: f64, angle: &Angle) -> f64 {
    let (value, settled) = wide_lane::<false>(x, angle);
    if settled {
        value
    } else {
        exact_float(x, angle)
    }
}

/// `x · C` narrowed from its float64 product with the constant's nearest
/// float64, and whether that is the float32 nearest the exact product.
///
/// The constant is within half a unit of its float64, the product rounds
/// once, so the float64 is within two of its units of the exact product:
/// more than eight from a float32 halfway point, the 2^28th unit past a
/// float32, and above float32's subnormals, both narrow alike. A zero is
/// exact; a NaN is left open.
#[inline(always)]
fn narrow_lane(x: f64, angle: &Angle) -> (f32, bool) {
    let product = x * angle.high;
    let past = product.to_bits() & ((1 << 29) - 1);
    let clear = past.abs_diff(1 << 28) > 8;
    let normal = product.abs() >= f64::from(f32::MIN_POSITIVE);
    (product as f32, (clear & normal) | (x == 0.0))
}

/// `x · C` from the constant as a double-double, and whether the bound
/// settles it: `x · high`, its exact error -- by a fused multiply-add where
/// `FMA` -- fused into `x · low`, and the two summed as a fast two-sum's
/// pair. The bound takes that rounding, or two without `FMA`, what the
/// halves leave of the constant, and the least normal float64 for the parts
/// rounded in the subnormal range, each by under the least subnormal -- a
/// subnormal constant costs [`wide_run_fma`] a microcode assist a vector.
/// A zero is exact, and keeps its sign.
#[inline(always)]
fn wide_lane<const FMA: bool>(x: f64, angle: &Angle) -> (f64, bool) {
    let product = x * angle.high;
    let low = if FMA {
        x.mul_add(angle.low, x.mul_add(angle.high, -product))
    } else {
        x * angle.low + product_error(x, angle.high, product)
    };
    let value = product + low;
    let rest = low - (value - product);
    let rounding = (x.abs() * angle.low.abs() + low.abs()) * 2f64.powi(-52);
    let bound = (rounding + x.abs() * angle.left) * (1.0 + 2f64.powi(-50)) + f64::MIN_POSITIVE;
    let (rounded, settled) = <f64 as Output>::settle_lanes(value, rest, bound);
    // A zero is the product itself, which keeps its sign where the sum of a
    // zero and its error's zero does not.
    let zero = x == 0.0;
    (if zero { product } else { rounded }, settled | zero)
}

/// [`narrow_lane`] over a run into `out`, in 256-bit registers where the
/// CPU has AVX2: whether any was left open.
fn narrow_run(x: &[f32], out: &mut [MaybeUninit<f32>], angle: &Angle) -> bool {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2 {
        // SAFETY: avx2 was detected on this CPU.
        return unsafe { narrow_run_avx2(x, out, angle) };
    }
    let mut open = false;
    for (slot, &x) in out.iter_mut().zip(x) {
        let (value, settled) = narrow_lane(f64::from(x), angle);
        open |= !settled;
        slot.write(value);
    }
    open
}

/// [`wide_lane`] over a run into `out`, with fused multiply-adds in 256-bit
/// registers where the CPU has them: whether any was left open.
fn wide_run(x: &[f64], out: &mut [MaybeUninit<f64>], angle: &Angle) -> bool {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { wide_run_fma(x, out, angle) };
    }
    let mut open = false;
    for (slot, &x) in out.iter_mut().zip(x) {
        let (value, settled) = wide_lane::<false>(x, angle);
        open |= !settled;
        slot.write(value);
    }
    open
}

/// [`narrow_lane`] in 256-bit registers, four float32s widened at once. The
/// halfway test is one comparison: `|past - 2^28| > 8` is `past - (2^28 -
/// 8) > 16` unsigned, compared signed with the sign bit turned over.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn narrow_run_avx2(x: &[f32], out: &mut [MaybeUninit<f32>], angle: &Angle) -> bool {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _CMP_GE_OQ, _mm_loadu_ps, _mm_storeu_ps, _mm256_and_si256, _mm256_andnot_pd,
        _mm256_castpd_si256, _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64,
        _mm256_cvtpd_ps, _mm256_cvtps_pd, _mm256_mul_pd, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_setzero_si256, _mm256_sub_epi64,
        _mm256_testc_si256, _mm256_xor_si256,
    };
    let high = _mm256_set1_pd(angle.high);
    let past = _mm256_set1_epi64x((1 << 29) - 1);
    let turned = i64::MIN;
    let (start, limit) = (
        _mm256_set1_epi64x((1 << 28) - 8),
        _mm256_set1_epi64x(16 ^ turned),
    );
    let flip = _mm256_set1_epi64x(turned);
    let (sign, zero) = (_mm256_set1_pd(-0.0), _mm256_setzero_pd());
    let least = _mm256_set1_pd(f64::from(f32::MIN_POSITIVE));
    let ones = _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256());
    // Every lane settled so far, all ones.
    let mut settled_all = ones;
    let (chunks, chunks_rest) = x.as_chunks::<4>();
    let (slots, slots_rest) = out.as_chunks_mut::<4>();
    for (chunk, slot) in chunks.iter().zip(slots.iter_mut()) {
        // SAFETY: each chunk holds four values.
        let x = _mm256_cvtps_pd(unsafe { _mm_loadu_ps(chunk.as_ptr()) });
        let product = _mm256_mul_pd(x, high);
        let bits = _mm256_and_si256(_mm256_castpd_si256(product), past);
        let shifted = _mm256_xor_si256(_mm256_sub_epi64(bits, start), flip);
        let clear = _mm256_cmpgt_epi64(shifted, limit);
        let normal = _mm256_castpd_si256(_mm256_cmp_pd(
            _mm256_andnot_pd(sign, product),
            least,
            _CMP_GE_OQ,
        ));
        let exact = _mm256_castpd_si256(_mm256_cmp_pd(x, zero, _CMP_EQ_OQ));
        let settled = _mm256_or_si256(_mm256_and_si256(clear, normal), exact);
        settled_all = _mm256_and_si256(settled_all, settled);
        // SAFETY: each chunk of the output holds four values.
        unsafe { _mm_storeu_ps(slot.as_mut_ptr().cast(), _mm256_cvtpd_ps(product)) };
    }
    let mut any = _mm256_testc_si256(settled_all, ones) == 0;
    for (slot, &x) in slots_rest.iter_mut().zip(chunks_rest) {
        let (value, settled) = narrow_lane(f64::from(x), angle);
        any |= !settled;
        slot.write(value);
    }
    any
}

/// [`wide_lane`] in 256-bit registers with fused multiply-adds, settled
/// more cheaply: with `rest` the rest of the pair after `x · high`, the
/// answer is the float64 `product + (rest ± bound)` rounds to where both
/// signs give the same one, rounding being monotone. `rest` is within
/// `|x| (|low| + 2^-53 high)` and a rounding, so the bound -- its rounding
/// and those of `rest ± bound`, and what the halves leave of the constant --
/// is `|x|` times one slope, and the least normal float64 for the parts
/// rounded in the subnormal range.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn wide_run_fma(x: &[f64], out: &mut [MaybeUninit<f64>], angle: &Angle) -> bool {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _mm256_add_pd, _mm256_andnot_pd, _mm256_blendv_pd, _mm256_cmp_pd,
        _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_mul_pd,
        _mm256_or_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let (high, low) = (_mm256_set1_pd(angle.high), _mm256_set1_pd(angle.low));
    let (sign, zero) = (_mm256_set1_pd(-0.0), _mm256_setzero_pd());
    let rest = angle.low.abs() + 2f64.powi(-53) * angle.high;
    let slope =
        (rest * 2f64.powi(-52) * (1.0 + 2f64.powi(-49)) + angle.left) * (1.0 + 2f64.powi(-50));
    let (slope, least) = (_mm256_set1_pd(slope), _mm256_set1_pd(f64::MIN_POSITIVE));
    // Any lane left open, as a sign bit.
    let mut open = zero;
    let (chunks, chunks_rest) = x.as_chunks::<4>();
    let (slots, slots_rest) = out.as_chunks_mut::<4>();
    for (chunk, slot) in chunks.iter().zip(slots.iter_mut()) {
        // SAFETY: each chunk holds four values.
        let x = unsafe { _mm256_loadu_pd(chunk.as_ptr()) };
        let product = _mm256_mul_pd(x, high);
        let rest = _mm256_fmadd_pd(x, low, _mm256_fmsub_pd(x, high, product));
        let bound = _mm256_fmadd_pd(_mm256_andnot_pd(sign, x), slope, least);
        let up = _mm256_add_pd(product, _mm256_add_pd(rest, bound));
        let down = _mm256_add_pd(product, _mm256_sub_pd(rest, bound));
        let zeros = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
        let settled = _mm256_or_pd(_mm256_cmp_pd(up, down, _CMP_EQ_OQ), zeros);
        open = _mm256_or_pd(open, _mm256_andnot_pd(settled, sign));
        // A zero is the product itself, with its sign.
        let value = _mm256_blendv_pd(up, product, zeros);
        // SAFETY: each chunk of the output holds four values.
        unsafe { _mm256_storeu_pd(slot.as_mut_ptr().cast(), value) };
    }
    let mut any = _mm256_movemask_pd(open) != 0;
    for (slot, &x) in slots_rest.iter_mut().zip(chunks_rest) {
        let (value, settled) = wide_lane::<true>(x, angle);
        any |= !settled;
        slot.write(value);
    }
    any
}

/// The value nearest `x · C` for a float `x`, by [`exact_scaled`]; a NaN or
/// an infinity as the product gives it.
#[cold]
#[inline(never)]
fn exact_float<T: Output>(x: f64, angle: &Angle) -> T {
    match Exact::float(x) {
        Some(exact) => exact_scaled(exact, angle),
        None => T::narrow(x * angle.high),
    }
}

/// The value nearest `X · 2^e · C` for the mantissa and exponent of
/// `exact`: `X` times the constant's 256 bits brackets `X · C` between
/// `X · bits` and `X · (bits + 1)`, both rounded -- the lower with a fraction
/// above it, the upper as the whole number below it with one -- and the two
/// agree; see the module's notes.
fn exact_scaled<T: Output>(exact: Exact, angle: &Angle) -> T {
    if exact.magnitude == 0 {
        return T::narrow(if exact.negative { -0.0 } else { 0.0 });
    }
    let mut product = product_of(&[exact.magnitude], &angle.bits);
    let least = exact.exponent - angle.shift;
    let lower: T = rounded_at(exact.negative, &product, true, least);
    let mut carry = u128::from(exact.magnitude - 1);
    for limb in product.iter_mut() {
        if carry == 0 {
            break;
        }
        let sum = u128::from(*limb) + carry;
        *limb = sum as u64;
        carry = sum >> 64;
    }
    let upper: T = rounded_at(exact.negative, &product, true, least);
    debug_assert!(
        lower == upper,
        "the bracket of an angle rounded apart, which the constants rule out"
    );
    lower
}

/// The gradient of `x · C`, which is the incoming gradient scaled as the
/// input was.
struct AngleBackward {
    angle: &'static Angle,
    input_ids: [TensorId; 1],
}

impl GradientFunction for AngleBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        accumulate_grad(
            &mut gradients,
            self.input_ids[0],
            scaled(grad_output, self.angle)?,
        )?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants' halves against their 256 bits, which they agree with
    /// to within what `left` allows.
    #[test]
    fn constants_agree() {
        for angle in [&TO_RADIANS, &TO_DEGREES] {
            let exact: f64 = exact_scaled(Exact::float(1.0).unwrap(), angle);
            assert_eq!(exact, angle.high);
            assert!(angle.left < angle.high * 2f64.powi(-100));
            assert!(angle.low.abs() <= angle.high * 2f64.powi(-52));
        }
    }

    #[test]
    fn exact_answers() {
        assert_eq!(wide_value(180.0, &TO_RADIANS), std::f64::consts::PI);
        assert_eq!(wide_value(std::f64::consts::PI, &TO_DEGREES), 180.0);
        assert_eq!(narrow_value(180.0, &TO_RADIANS), std::f32::consts::PI);
        assert_eq!(wide_value(-0.0, &TO_RADIANS).to_bits(), (-0.0f64).to_bits());
        assert!(wide_value(f64::NAN, &TO_RADIANS).is_nan());
        assert_eq!(wide_value(f64::MAX, &TO_DEGREES), f64::INFINITY);
        assert_eq!(
            wide_value(f64::NEG_INFINITY, &TO_RADIANS),
            f64::NEG_INFINITY
        );
    }
}
