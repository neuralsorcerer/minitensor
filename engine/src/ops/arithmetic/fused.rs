// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `lerp`, `addcmul` and `addcdiv`, each element rounded once.
//!
//! Written as their formulas, each rounded at every step: `lerp` at the
//! difference, the product and the sum, so that hardly a run of ordinary
//! data came out without an element an ulp or more out, and `weight = 1`
//! did not give `end` where the difference rounded. Each element is now the
//! float nearest the exact value; see `exact_sum::lerp_of`, `addcmul_of`
//! and `addcdiv_of`.

use std::mem::MaybeUninit;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::exact_sum::{
    Element, Output, addcdiv_lane, addcdiv_lane_narrow, addcdiv_of, addcmul_lane,
    addcmul_lane_narrow, addcmul_of, lerp_lane, lerp_lane_narrow, lerp_of,
};
use crate::{
    autograd::{GradientFunction, TensorId, accumulate_grad, with_grad_fn},
    error::{MinitensorError, Result},
    ops::arithmetic,
    ops::binary::{promote_arithmetic_dtype, promote_division_dtype},
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// The point `weight` of the way from `start` to `end`,
/// `start + weight · (end - start)`, the three broadcast together, each
/// element the float nearest the exact value: `weight = 0` gives `start`
/// and `weight = 1` gives `end`.
pub fn lerp(start: &Tensor, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
    fused(Fused::Lerp, [start, end, weight], 0.0)
}

/// `input + value · tensor1 · tensor2`, broadcast together, each element the
/// float nearest the exact value. `value` is read at the result's width.
pub fn addcmul(input: &Tensor, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
    fused(Fused::Addcmul, [input, tensor1, tensor2], value)
}

/// `input + value · tensor1 / tensor2`, broadcast together, each element the
/// float nearest the exact value. `value` is read at the result's width.
pub fn addcdiv(input: &Tensor, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
    fused(Fused::Addcdiv, [input, tensor1, tensor2], value)
}

/// `input + alpha · other`, broadcast together, each element the float
/// nearest the exact value, `alpha` read at the result's width as a literal
/// beside it is.
///
/// Two float operands of one dtype take a kernel of their own, a single
/// element read as it is and any other broadcast spread first: a fused
/// multiply-add for float64, which rounds the exact value once, and for
/// float32 the float64 sum, whose product is exact, rounded to odd at
/// float64's last bit by its two-sum's error and narrowed. Mixed operands
/// take `addcmul`'s kernel with a unit factor, which reads an integer
/// exactly too.
pub fn add_scaled(input: &Tensor, other: &Tensor, alpha: f64) -> Result<Tensor> {
    let dtype = input.dtype();
    let floats = dtype == other.dtype()
        && matches!(dtype, DataType::Float32 | DataType::Float64)
        && input.device() == other.device();
    if !floats {
        // A bool one promotes to whatever the other two do.
        let one = Tensor::new(
            Arc::new(TensorData::from_vec(
                vec![true],
                DataType::Bool,
                input.device(),
            )),
            Shape::scalar(),
            DataType::Bool,
            input.device(),
            false,
        );
        return fused(Fused::Addcmul, [input, other, &one], alpha);
    }
    let shape = input.shape().broadcast_with(other.shape())?;
    let dims: Vec<isize> = shape.dims().iter().map(|&size| size as isize).collect();
    let spread = |tensor: &Tensor| -> Result<Tensor> {
        let tensor = tensor.detach();
        if tensor.shape() == &shape {
            tensor.contiguous()
        } else {
            tensor.expand(dims.clone())?.contiguous()
        }
    };
    let x = spread(input)?;
    // `other` read as it is where its elements repeat with its length as the
    // period: a single element, or the trailing axes of the result.
    let leading = other
        .shape()
        .dims()
        .iter()
        .take_while(|&&size| size == 1)
        .count();
    let tail = &other.shape().dims()[leading..];
    let periodic = tail.len() <= shape.ndim()
        && tail == &shape.dims()[shape.ndim() - tail.len()..]
        && shape.numel() > 0;
    let y = if periodic {
        other.detach().contiguous()?
    } else {
        spread(other)?
    };
    let missing = || MinitensorError::internal_error("add_scaled: a float slice");
    let (data, alpha) = if dtype == DataType::Float32 {
        let alpha = f64::from(alpha as f32);
        let (a, b) = (x.data().as_f32_slice(), y.data().as_f32_slice());
        let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
        let out = scaled_sums(a, b, |x, y, out| narrow_scaled_run(x, y, alpha, out));
        (TensorData::from_vec(out, dtype, x.device()), alpha)
    } else {
        let (a, b) = (x.data().as_f64_slice(), y.data().as_f64_slice());
        let (a, b) = (a.ok_or_else(missing)?, b.ok_or_else(missing)?);
        let out = scaled_sums(a, b, |x, y, out| wide_scaled_run(x, y, alpha, out));
        (TensorData::from_vec(out, dtype, x.device()), alpha)
    };
    let requires_grad =
        (input.requires_grad() || other.requires_grad()) && crate::autograd::is_grad_enabled();
    let output = Tensor::new(Arc::new(data), shape, dtype, x.device(), requires_grad);
    if !requires_grad {
        return Ok(output);
    }
    let grad_fn = Arc::new(AddScaledBackward {
        alpha,
        shapes: [input.shape().clone(), other.shape().clone()],
        input_ids: [input.id(), other.id()],
        input_requires_grad: [input.requires_grad(), other.requires_grad()],
    });
    with_grad_fn(output, grad_fn)
}

/// `run` over `x` and `y` a block to a task, across the pool for a long
/// operand: `y` as long as `x`, a single element, or repeating with its
/// length as the period, read in runs that line up with it -- a period
/// under a thousand elements repeated into one at least that long first, so
/// the runs are not short.
fn scaled_sums<I: Copy + Sync, T: Copy + Send + Sync>(
    x: &[I],
    y: &[I],
    run: impl Fn(&[I], &[I], &mut [MaybeUninit<T>]) + Sync,
) -> Vec<T> {
    let tile: Vec<I>;
    let y = if y.len() > 1 && y.len() < 1024 && y.len() < x.len() {
        tile = y
            .iter()
            .copied()
            .cycle()
            .take(1024usize.div_ceil(y.len()) * y.len())
            .collect();
        &tile[..]
    } else {
        y
    };
    let fill = |start: usize, block: &mut [MaybeUninit<T>]| {
        let end = start + block.len();
        if y.len() == 1 {
            return run(&x[start..end], y, block);
        }
        let period = y.len();
        let mut at = start;
        for out in block.chunks_mut(period) {
            // A run from `at` up to the next multiple of the period, or the
            // block's end.
            let offset = at % period;
            let len = (period - offset).min(end - at);
            let (now, later) = out.split_at_mut(len.min(out.len()));
            run(&x[at..at + now.len()], &y[offset..offset + now.len()], now);
            at += now.len();
            if !later.is_empty() {
                let rest = later.len();
                run(&x[at..at + rest], &y[..rest], later);
                at += rest;
            }
        }
    };
    // SAFETY: the chunks cover the output, and `fill` writes every element
    // of each.
    unsafe {
        crate::ops::map::build_vec(x.len(), |spare| {
            crate::ops::map::par_out_chunks(spare, 1 << 14, &fill)
        })
    }
}

/// Each `x + alpha · y` of float64s, by a fused multiply-add: in 256-bit
/// registers where the CPU has them, and from the library's otherwise, which
/// rounds once as well.
fn wide_scaled_run(x: &[f64], y: &[f64], alpha: f64, out: &mut [MaybeUninit<f64>]) {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn fused_run(x: &[f64], y: &[f64], alpha: f64, out: &mut [MaybeUninit<f64>]) {
        wide_scaled_lanes(x, y, alpha, out);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { fused_run(x, y, alpha, out) };
    }
    wide_scaled_lanes(x, y, alpha, out);
}

#[inline(always)]
fn wide_scaled_lanes(x: &[f64], y: &[f64], alpha: f64, out: &mut [MaybeUninit<f64>]) {
    if let [y] = y {
        for (slot, &x) in out.iter_mut().zip(x) {
            slot.write(alpha.mul_add(*y, x));
        }
    } else {
        for ((slot, &x), &y) in out.iter_mut().zip(x).zip(y) {
            slot.write(alpha.mul_add(y, x));
        }
    }
}

/// Each `x + alpha · y` of float32s, `alpha` a float32 too, in 256-bit
/// registers where the CPU has AVX2; see [`narrow_scaled`].
fn narrow_scaled_run(x: &[f32], y: &[f32], alpha: f64, out: &mut [MaybeUninit<f32>]) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2 {
        // SAFETY: avx2 was detected on this CPU.
        return unsafe { narrow_scaled_avx2(x, y, alpha, out) };
    }
    let y_at = |index: usize| if y.len() == 1 { y[0] } else { y[index] };
    for (index, (slot, &x)) in out.iter_mut().zip(x).enumerate() {
        slot.write(narrow_scaled(f64::from(x), f64::from(y_at(index)), alpha));
    }
}

/// The float32 nearest `x + alpha · y` for float32 operands: the product is
/// exact in float64 and the sum rounds once there, so the two-sum's pair is
/// the exact sum. Rounded to odd at float64's last bit -- the sum where it
/// is exact or odd, and otherwise its neighbour towards the error -- it
/// narrows to the float32 nearest the exact sum, subnormals included.
#[inline(always)]
fn narrow_scaled(x: f64, y: f64, alpha: f64) -> f32 {
    let (sum, error) = super::rounded_once::two_sum(x, alpha * y);
    let bits = sum.to_bits();
    // A NaN error, beside a NaN or infinite sum, moves nothing.
    let odd = if error != 0.0 && !error.is_nan() && bits & 1 == 0 {
        if (error > 0.0) == (sum > 0.0) {
            bits + 1
        } else {
            bits - 1
        }
    } else {
        bits
    };
    f64::from_bits(odd) as f32
}

/// [`narrow_scaled`] in 256-bit registers, four float32s of each operand
/// widened at once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn narrow_scaled_avx2(x: &[f32], y: &[f32], alpha: f64, out: &mut [MaybeUninit<f32>]) {
    use std::arch::x86_64::{
        _CMP_NEQ_OQ, _mm_loadu_ps, _mm_set1_ps, _mm_storeu_ps, _mm256_add_epi64, _mm256_add_pd,
        _mm256_and_si256, _mm256_blendv_epi8, _mm256_castpd_si256, _mm256_castsi256_pd,
        _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_cvtpd_ps, _mm256_cvtps_pd, _mm256_mul_pd,
        _mm256_set1_epi64x, _mm256_set1_pd, _mm256_setzero_pd, _mm256_setzero_si256, _mm256_sub_pd,
        _mm256_xor_si256,
    };
    let scale = _mm256_set1_pd(alpha);
    let zero = _mm256_setzero_pd();
    let (one, minus_one) = (_mm256_set1_epi64x(1), _mm256_set1_epi64x(-1));
    let (sign, nothing) = (_mm256_set1_epi64x(i64::MIN), _mm256_setzero_si256());
    let single = y.len() == 1;
    let constant = if single { y[0] } else { 0.0 };
    let (chunks, chunks_rest) = x.as_chunks::<4>();
    let (slots, slots_rest) = out.as_chunks_mut::<4>();
    for (nth, (chunk, slot)) in chunks.iter().zip(slots.iter_mut()).enumerate() {
        // SAFETY: each chunk holds four values, and `y`, unless a single
        // element, is as long as `x`.
        let (a, b) = unsafe {
            let b = if single {
                _mm_set1_ps(constant)
            } else {
                _mm_loadu_ps(y.as_ptr().add(4 * nth))
            };
            (
                _mm256_cvtps_pd(_mm_loadu_ps(chunk.as_ptr())),
                _mm256_cvtps_pd(b),
            )
        };
        let product = _mm256_mul_pd(scale, b);
        // The sum and its exact error, by a two-sum.
        let sum = _mm256_add_pd(a, product);
        let virtual_product = _mm256_sub_pd(sum, a);
        let virtual_input = _mm256_sub_pd(sum, virtual_product);
        let error = _mm256_add_pd(
            _mm256_sub_pd(a, virtual_input),
            _mm256_sub_pd(product, virtual_product),
        );
        // Rounded to odd: an even sum with an error moves one unit towards
        // it -- away from zero where the two share a sign.
        let bits = _mm256_castpd_si256(sum);
        let even = _mm256_cmpeq_epi64(_mm256_and_si256(bits, one), nothing);
        let moved = _mm256_castpd_si256(_mm256_cmp_pd(error, zero, _CMP_NEQ_OQ));
        let apart = _mm256_cmpeq_epi64(
            _mm256_and_si256(_mm256_xor_si256(_mm256_castpd_si256(error), bits), sign),
            sign,
        );
        let step = _mm256_blendv_epi8(one, minus_one, apart);
        let odd = _mm256_add_epi64(bits, _mm256_and_si256(step, _mm256_and_si256(even, moved)));
        // SAFETY: each chunk of the output holds four values.
        unsafe {
            _mm_storeu_ps(
                slot.as_mut_ptr().cast(),
                _mm256_cvtpd_ps(_mm256_castsi256_pd(odd)),
            )
        };
    }
    let start = 4 * chunks.len();
    for (offset, (slot, &x)) in slots_rest.iter_mut().zip(chunks_rest).enumerate() {
        let y = if single { y[0] } else { y[start + offset] };
        slot.write(narrow_scaled(f64::from(x), f64::from(y), alpha));
    }
}

/// The gradient of `input + alpha · other`: the incoming one for `input`,
/// and `alpha` times it for `other`, each summed to its operand's shape.
struct AddScaledBackward {
    alpha: f64,
    shapes: [Shape; 2],
    input_ids: [TensorId; 2],
    input_requires_grad: [bool; 2],
}

impl GradientFunction for AddScaledBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        if self.input_requires_grad[0] {
            let term =
                crate::autograd::reduce_gradient_for_broadcasting(grad_output, &self.shapes[0])?;
            accumulate_grad(&mut gradients, self.input_ids[0], term)?;
        }
        if self.input_requires_grad[1] {
            let alpha = crate::ops::util::create_scalar_tensor(
                self.alpha,
                grad_output.dtype(),
                grad_output.device(),
            )?;
            let term = arithmetic::mul(grad_output, &alpha)?;
            let term = crate::autograd::reduce_gradient_for_broadcasting(&term, &self.shapes[1])?;
            accumulate_grad(&mut gradients, self.input_ids[1], term)?;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Which of the three, with `value` read at the result's width.
#[derive(Clone, Copy)]
enum Fused {
    Lerp,
    Addcmul,
    Addcdiv,
}

impl Fused {
    fn name(self) -> &'static str {
        match self {
            Fused::Lerp => "lerp",
            Fused::Addcmul => "addcmul",
            Fused::Addcdiv => "addcdiv",
        }
    }
}

/// The three operands promoted as the formula's operators promote them and
/// broadcast together. All float32 with a float32 result are read as they
/// are; anything else is read in float64, which holds every float32 and
/// int32 exactly, and the result rounded once to its dtype -- an int32
/// beside a float32 is not rounded to float32 first.
fn fused(kind: Fused, operands: [&Tensor; 3], value: f64) -> Result<Tensor> {
    let [a, b, c] = operands;
    if a.device() != b.device() || a.device() != c.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", a.device()),
            format!(
                "{:?}",
                if a.device() != b.device() { b } else { c }.device()
            ),
        ));
    }
    // As the formula's operators promote: `tensor1 / tensor2` is a float even
    // for two integers.
    let dtype = match kind {
        Fused::Addcdiv => {
            promote_arithmetic_dtype(a.dtype(), promote_division_dtype(b.dtype(), c.dtype()))
        }
        _ => promote_arithmetic_dtype(promote_arithmetic_dtype(a.dtype(), b.dtype()), c.dtype()),
    };
    if !matches!(dtype, DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(format!(
            "{} takes operands that promote to a float dtype, got {:?}, {:?} and {:?}",
            kind.name(),
            a.dtype(),
            b.dtype(),
            c.dtype()
        )));
    }
    let shape = a
        .shape()
        .broadcast_with(b.shape())?
        .broadcast_with(c.shape())?;
    let narrow = dtype == DataType::Float32 && operands.iter().all(|t| t.dtype() == dtype);
    let read_as = if narrow {
        DataType::Float32
    } else {
        DataType::Float64
    };
    let cast = [a.astype(read_as)?, b.astype(read_as)?, c.astype(read_as)?];
    let value = if dtype == DataType::Float32 {
        f64::from(value as f32)
    } else {
        value
    };
    let (narrow_slices, wide_slices) = (TensorData::as_f32_slice, TensorData::as_f64_slice);
    let data = match (read_as, dtype) {
        (DataType::Float32, _) => {
            evaluate::<f32, f32>(kind, &cast, narrow_slices, &shape, value, true)?
        }
        (_, DataType::Float32) => {
            evaluate::<f64, f32>(kind, &cast, wide_slices, &shape, value, false)?
        }
        _ => evaluate::<f64, f64>(kind, &cast, wide_slices, &shape, value, false)?,
    };
    let requires_grad =
        cast.iter().any(|operand| operand.requires_grad()) && crate::autograd::is_grad_enabled();
    let output = Tensor::new(Arc::new(data), shape, dtype, a.device(), requires_grad);
    if !requires_grad {
        return Ok(output);
    }
    let [a, b, c] = &cast;
    let grad_fn = Arc::new(FusedBackward {
        kind,
        value,
        input_ids: [a.id(), b.id(), c.id()],
        input_requires_grad: [a.requires_grad(), b.requires_grad(), c.requires_grad()],
        operands: [a.detach(), b.detach(), c.detach()],
    });
    with_grad_fn(output, grad_fn)
}

/// Whether a fused multiply-add is an instruction on every CPU of the
/// target, rather than one to look for at run time.
const NATIVE_FMA: bool = cfg!(target_arch = "aarch64");

/// The output of `kind` over `operands`, read as `I` by `read` and rounded
/// to `T`, a block to a task, with fused multiply-adds where the CPU has
/// them; `narrow` when the operands are float32, whose lanes need no split.
fn evaluate<I: Element, T: Output>(
    kind: Fused,
    operands: &[Tensor; 3],
    read: fn(&TensorData) -> Option<&[I]>,
    shape: &Shape,
    value: f64,
    narrow: bool,
) -> Result<TensorData> {
    let walk = Walk::<I, 3>::new(operands, read, shape.dims())?;
    macro_rules! run {
        ($lane:ident, $full:ident) => {{
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            fn fill_fma<I: Element, T: Output>(
                walk: &Walk<'_, I, 3>,
                block: &mut [MaybeUninit<T>],
                start: usize,
                value: f64,
            ) {
                walk.fill(
                    block,
                    start,
                    |[x, y, z]| $lane::<T, true>(x, y, z, value),
                    |[x, y, z]| $full::<T, true>(x, y, z, value),
                );
            }
            let fill = |start: usize, block: &mut [MaybeUninit<T>]| {
                #[cfg(target_arch = "x86_64")]
                if crate::ops::simd::simd_capabilities().avx2_fma {
                    // SAFETY: avx2 and fma were detected on this CPU.
                    return unsafe { fill_fma(&walk, block, start, value) };
                }
                walk.fill(
                    block,
                    start,
                    |[x, y, z]| $lane::<T, NATIVE_FMA>(x, y, z, value),
                    |[x, y, z]| $full::<T, NATIVE_FMA>(x, y, z, value),
                );
            };
            // SAFETY: the chunks cover the output, and `fill` writes every
            // element of each.
            unsafe {
                crate::ops::map::build_vec(shape.numel(), |spare| {
                    crate::ops::map::par_out_chunks(spare, 1 << 14, &fill)
                })
            }
        }};
    }
    let out: Vec<T> = match (kind, narrow) {
        (Fused::Lerp, false) => run!(lerp_lane_kernel, lerp_kernel),
        (Fused::Addcmul, false) => run!(addcmul_lane, addcmul_of),
        (Fused::Addcdiv, false) => run!(addcdiv_lane, addcdiv_of),
        (Fused::Lerp, true) => run!(lerp_lane_narrow_kernel, lerp_kernel),
        (Fused::Addcmul, true) => run!(addcmul_lane_narrow_kernel, addcmul_of),
        (Fused::Addcdiv, true) => run!(addcdiv_lane_narrow_kernel, addcdiv_of),
    };
    Ok(TensorData::from_vec(out, T::DTYPE, operands[0].device()))
}

/// [`lerp_of`] in the shape of the other two kernels, with no `value`.
#[inline(always)]
fn lerp_kernel<T: Output, const FMA: bool>(start: f64, end: f64, weight: f64, _value: f64) -> T {
    lerp_of::<T, FMA>(start, end, weight)
}

/// [`lerp_lane`] in the shape of the other two lanes, with no `value`.
#[inline(always)]
fn lerp_lane_kernel<T: Output, const FMA: bool>(
    start: f64,
    end: f64,
    weight: f64,
    _value: f64,
) -> (T, bool) {
    lerp_lane::<T, FMA>(start, end, weight)
}

/// Elements a [`Walk`] gathers and computes at a time.
const RUN: usize = 256;

/// The float32 lanes in the shape of the others, which take whether to fuse.
#[inline(always)]
fn lerp_lane_narrow_kernel<T: Output, const FMA: bool>(
    start: f64,
    end: f64,
    weight: f64,
    _value: f64,
) -> (T, bool) {
    lerp_lane_narrow::<T>(start, end, weight)
}

/// See [`lerp_lane_narrow_kernel`].
#[inline(always)]
fn addcmul_lane_narrow_kernel<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    addcmul_lane_narrow::<T>(input, first, second, value)
}

/// See [`lerp_lane_narrow_kernel`].
#[inline(always)]
fn addcdiv_lane_narrow_kernel<T: Output, const FMA: bool>(
    input: f64,
    first: f64,
    second: f64,
    value: f64,
) -> (T, bool) {
    addcdiv_lane_narrow::<T>(input, first, second, value)
}

/// `N` operands of one element type read in the order of the output they
/// broadcast to.
struct Walk<'a, I, const N: usize> {
    values: [&'a [I]; N],
    dims: &'a [usize],
    /// Each operand's step along every output axis, zero along one it is
    /// broadcast over.
    steps: [Vec<usize>; N],
    /// Whether every operand is the output's length or a single element,
    /// so that each is read straight from its slice.
    flat: bool,
}

impl<'a, I: Element, const N: usize> Walk<'a, I, N> {
    fn new(
        operands: &'a [Tensor; N],
        read: fn(&TensorData) -> Option<&[I]>,
        dims: &'a [usize],
    ) -> Result<Self> {
        let mut values: [&[I]; N] = [&[]; N];
        for (slot, operand) in values.iter_mut().zip(operands) {
            *slot = read(operand.data()).ok_or_else(|| {
                MinitensorError::internal_error("an operand without the slice it was cast to")
            })?;
        }
        let steps = std::array::from_fn(|k| {
            let shape = operands[k].shape().dims();
            let mut steps = vec![0; dims.len()];
            let mut step = 1;
            for (axis, &size) in shape.iter().enumerate().rev() {
                if size != 1 {
                    steps[dims.len() - shape.len() + axis] = step;
                }
                step *= size;
            }
            steps
        });
        let len = dims.iter().product::<usize>();
        let flat = values
            .iter()
            .all(|values| values.len() == len || values.len() == 1);
        Ok(Self {
            values,
            dims,
            steps,
            flat,
        })
    }

    /// `block` of the output from element `start` on, a run of elements at
    /// a time: `lane` over the run without a branch, and `full` again for
    /// the elements whose lane did not settle. Operands of the output's
    /// length are read in place and single elements from a run of copies;
    /// any other broadcast is gathered into float64 by an odometer first.
    #[inline(always)]
    fn fill<T: Copy>(
        &self,
        block: &mut [MaybeUninit<T>],
        start: usize,
        lane: impl Fn([f64; N]) -> (T, bool),
        full: impl Fn([f64; N]) -> T,
    ) {
        if self.flat {
            let copies: [[I; RUN]; N] = std::array::from_fn(|k| [self.values[k][0]; RUN]);
            let mut first = start;
            for run in block.chunks_mut(RUN) {
                let count = run.len();
                let sources: [&[I]; N] = std::array::from_fn(|k| {
                    if self.values[k].len() == 1 {
                        &copies[k][..count]
                    } else {
                        &self.values[k][first..first + count]
                    }
                });
                compute(sources, run, &lane, &full);
                first += count;
            }
            return;
        }
        let mut gathered = [[0f64; RUN]; N];
        let mut odometer = Odometer::at(self, start);
        for run in block.chunks_mut(RUN) {
            let count = run.len();
            for j in 0..count {
                for k in 0..N {
                    gathered[k][j] = self.values[k][odometer.at[k]].wide();
                }
                odometer.turn(self);
            }
            let sources: [&[f64]; N] = std::array::from_fn(|k| &gathered[k][..count]);
            compute(sources, run, &lane, &full);
        }
    }
}

/// `run[j]`, from element `j` of each of `sources`: `lane` over the run
/// without a branch, then `full` for the elements whose lane did not settle.
/// Every element of `run` is written.
#[inline(always)]
fn compute<S: Element, T: Copy, const N: usize>(
    sources: [&[S]; N],
    run: &mut [MaybeUninit<T>],
    lane: &impl Fn([f64; N]) -> (T, bool),
    full: &impl Fn([f64; N]) -> T,
) {
    let count = run.len();
    let mut unsettled = [false; RUN];
    let mut any = false;
    for j in 0..count {
        let (value, settled) = lane(std::array::from_fn(|k| sources[k][j].wide()));
        run[j].write(value);
        unsettled[j] = !settled;
        any |= !settled;
    }
    if any {
        for j in 0..count {
            if unsettled[j] {
                run[j].write(full(std::array::from_fn(|k| sources[k][j].wide())));
            }
        }
    }
}

/// The index of an output element along every axis, and in each operand.
struct Odometer<const N: usize> {
    index: Vec<usize>,
    at: [usize; N],
}

impl<const N: usize> Odometer<N> {
    /// At output element `start` of `walk`.
    fn at<I>(walk: &Walk<'_, I, N>, start: usize) -> Self {
        let rank = walk.dims.len();
        let mut index = vec![0; rank];
        let mut at = [0usize; N];
        let mut rest = start;
        for axis in (0..rank).rev() {
            if walk.dims[axis] == 0 {
                break;
            }
            index[axis] = rest % walk.dims[axis];
            rest /= walk.dims[axis];
            for (at, steps) in at.iter_mut().zip(&walk.steps) {
                *at += index[axis] * steps[axis];
            }
        }
        Self { index, at }
    }

    /// On to the next output element.
    #[inline(always)]
    fn turn<I>(&mut self, walk: &Walk<'_, I, N>) {
        for axis in (0..walk.dims.len()).rev() {
            self.index[axis] += 1;
            for (at, steps) in self.at.iter_mut().zip(&walk.steps) {
                *at += steps[axis];
            }
            if self.index[axis] < walk.dims[axis] {
                return;
            }
            self.index[axis] = 0;
            for (at, steps) in self.at.iter_mut().zip(&walk.steps) {
                *at -= steps[axis] * walk.dims[axis];
            }
        }
    }
}

/// The gradient of the three, each partial derivative times the incoming
/// gradient and summed back over the axes its operand was broadcast along:
///
/// - `lerp`: `1 - weight`, `weight` and `end - start`;
/// - `addcmul`: `1`, `value · tensor2` and `value · tensor1`;
/// - `addcdiv`: `1`, `value / tensor2` and `-value · tensor1 / tensor2²`.
struct FusedBackward {
    kind: Fused,
    value: f64,
    operands: [Tensor; 3],
    input_ids: [TensorId; 3],
    input_requires_grad: [bool; 3],
}

impl GradientFunction for FusedBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let [a, b, c] = &self.operands;
        // The operands' own dtype, which is the output's or wider.
        let grad = grad_output.astype(a.dtype())?;
        let scalar =
            |value: f64| crate::ops::util::create_scalar_tensor(value, a.dtype(), a.device());
        let mut gradients = FxHashMap::default();
        for index in 0..3 {
            if !self.input_requires_grad[index] {
                continue;
            }
            let partial = match (self.kind, index) {
                (Fused::Lerp, 0) => arithmetic::sub(&scalar(1.0)?, c)?,
                (Fused::Lerp, 1) => c.clone(),
                (Fused::Lerp, _) => arithmetic::sub(b, a)?,
                (_, 0) => scalar(1.0)?,
                (Fused::Addcmul, 1) => arithmetic::mul(c, &scalar(self.value)?)?,
                (Fused::Addcmul, _) => arithmetic::mul(b, &scalar(self.value)?)?,
                (Fused::Addcdiv, 1) => arithmetic::div(&scalar(self.value)?, c)?,
                (Fused::Addcdiv, _) => {
                    let quotient = arithmetic::div(b, &arithmetic::mul(c, c)?)?;
                    arithmetic::mul(&quotient, &scalar(-self.value)?)?
                }
            };
            let term = arithmetic::mul(&grad, &partial)?;
            let shape = self.operands[index].shape();
            let term = crate::autograd::reduce_gradient_for_broadcasting(&term, shape)?;
            accumulate_grad(&mut gradients, self.input_ids[index], term)?;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
