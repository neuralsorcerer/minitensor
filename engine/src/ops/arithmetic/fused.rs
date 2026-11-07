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
