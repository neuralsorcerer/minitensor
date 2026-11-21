// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Running totals that are not sums.
//!
//! `cumsum` and `cumprod` already exist, built on `scan_along_dim` with `+`
//! and `*` as the step. These are the other three worth having: the running
//! maximum and minimum, which also report *where* they came from, and the
//! running log-sum-exp, which is the only way to accumulate probabilities over
//! a long axis without underflowing.
//!
//! None of them composes. A scan is a recurrence -- each step needs the
//! previous step's result -- and no arrangement of the elementwise, reduction
//! and contraction operations in this library runs one.
//!
//! ## Why `logcumsumexp` is not `cumsum` of `exp`
//!
//! That is what it means, and it is unusable. `exp` of a log-probability
//! underflows to zero after a few hundred steps, so the running total stops
//! moving and the answer becomes `-inf` for every position past that. Carrying
//! the running total beside its largest value -- as `Σ exp(x - max)`, a pair of
//! floats, see `rounded_prefixes` -- keeps every step representable, which is
//! the whole point of the operation existing separately.
//!
//! ## Why the extremum scan is not `scan_along_dim`
//!
//! It writes two outputs rather than one, and the second is a position.
//! `scan_along_dim` seeds with `out[i] = widen(inp[i])` -- a closure that never
//! sees the index it is seeding -- so a position cannot be introduced. Same
//! walk, different shape, which is a different computation rather than a
//! duplicated one.

use crate::{
    autograd::{CummaxBackward, LogcumsumexpBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        arithmetic::{
            exact_product::{NAN_KIND, kind_of, kind_times, sum_of_kinds},
            exact_sum::Output,
            extended::{self, Pair},
            rounded_once::two_sum,
        },
        map::{par_out_chunks, par_out_chunks2},
        util::normalize_dim,
    },
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// How a tensor is laid out for a scan along one axis: `outer` slabs, each of
/// `dim_size` rows of `inner` contiguous elements.
fn slab_of(tensor: &Tensor, dim: usize) -> (usize, usize) {
    let dims = tensor.shape().dims();
    let inner: usize = dims[dim + 1..].iter().product();
    (dims[dim], inner)
}

/// The running extremum along one axis, and the position it came from.
///
/// `better(candidate, running)` decides whether the candidate replaces what is
/// there. It must be strict, so that a tie leaves the earlier position standing.
fn scan_extreme<T, F>(
    values: &mut [T],
    indices: &mut [i64],
    input: &[T],
    dim_size: usize,
    inner: usize,
    better: F,
) where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> bool + Send + Sync,
{
    let slab = dim_size * inner;
    if slab == 0 {
        return;
    }
    par_out_chunks2(values, indices, slab, &|start, value_slab, index_slab| {
        let source = &input[start..start + value_slab.len()];
        if inner == 1 {
            // The scanned axis is the last one, so a slab is one running
            // extremum and one position. Carrying both in registers is the
            // difference between a scan and what the general form below
            // degenerates to here: two `split_at_mut` calls and two
            // one-iteration loops *per element*.
            let mut best = source[0];
            let mut at = 0i64;
            value_slab[0] = best;
            index_slab[0] = 0;
            for d in 1..dim_size {
                let candidate = source[d];
                if better(candidate, best) {
                    best = candidate;
                    at = d as i64;
                }
                value_slab[d] = best;
                index_slab[d] = at;
            }
            return;
        }
        for i in 0..inner {
            value_slab[i] = source[i];
            index_slab[i] = 0;
        }
        for d in 1..dim_size {
            let (previous, current) =
                value_slab[(d - 1) * inner..(d + 1) * inner].split_at_mut(inner);
            let (before, now) = index_slab[(d - 1) * inner..(d + 1) * inner].split_at_mut(inner);
            for i in 0..inner {
                let candidate = source[d * inner + i];
                if better(candidate, previous[i]) {
                    current[i] = candidate;
                    now[i] = d as i64;
                } else {
                    current[i] = previous[i];
                    now[i] = before[i];
                }
            }
        }
    });
}

/// Both extremum scans, which differ only in which way the comparison runs.
fn cumextreme(tensor: &Tensor, dim: isize, greater: bool) -> Result<(Tensor, Tensor)> {
    let dim = normalize_dim(dim, tensor.ndim())?;
    if tensor.dtype() == DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "cummax and cummin are not supported for boolean tensors",
        ));
    }
    let contiguous = tensor.contiguous()?;
    let (dim_size, inner) = slab_of(tensor, dim);
    let numel = tensor.numel();
    let device = tensor.device();

    let mut value_data = TensorData::zeros_on_device(numel, tensor.dtype(), device);
    let mut index_data = TensorData::zeros_on_device(numel, DataType::Int64, device);
    let positions = index_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("cummax: indices are not int64"))?;

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident, $ty:ty, $nan:expr) => {{
            let source = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("cummax: dtype does not match the input")
            })?;
            let out = value_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("cummax: dtype does not match the output")
            })?;
            // A NaN swallows the running extremum and keeps it -- the first one
            // wins, and nothing after it displaces it. Without the guard on
            // `running` a later NaN would move the index while changing nothing
            // about the value.
            let unordered: fn($ty) -> bool = $nan;
            let better = move |candidate: $ty, running: $ty| {
                !unordered(running)
                    && (unordered(candidate)
                        || if greater {
                            candidate > running
                        } else {
                            candidate < running
                        })
            };
            scan_extreme(out, positions, source, dim_size, inner, better);
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut, f32, |v| v.is_nan()),
        DataType::Float64 => run!(as_f64_slice, as_f64_slice_mut, f64, |v| v.is_nan()),
        DataType::Int32 => run!(as_i32_slice, as_i32_slice_mut, i32, |_| false),
        DataType::Int64 => run!(as_i64_slice, as_i64_slice_mut, i64, |_| false),
        DataType::Bool => unreachable!("rejected above"),
    }

    let shape = tensor.shape().clone();
    let indices = Tensor::new(
        Arc::new(index_data),
        shape.clone(),
        DataType::Int64,
        device,
        false,
    );
    let mut values = Tensor::new(
        Arc::new(value_data),
        shape,
        tensor.dtype(),
        device,
        tensor.requires_grad() && tensor.dtype().is_float(),
    );

    if values.requires_grad() {
        // Each output took its value from exactly one input position, so the
        // gradient goes there and nowhere else -- the same rule `max` follows,
        // applied once per prefix.
        let grad_fn = Arc::new(CummaxBackward {
            input_id: tensor.id(),
            indices: indices.detach(),
            dim,
        });
        values = with_grad_fn(values, grad_fn)?;
    }
    Ok((values, indices))
}

/// The running maximum along `dim`, and where each one came from.
///
/// Ties go to the earliest position holding the value, which is a choice rather
/// than a consequence -- and the same one `mode` makes here. A `NaN` takes over
/// the running maximum and keeps it, as it does for `max`.
pub fn cummax(tensor: &Tensor, dim: isize) -> Result<(Tensor, Tensor)> {
    cumextreme(tensor, dim, true)
}

/// The running minimum along `dim`, and where each one came from.
pub fn cummin(tensor: &Tensor, dim: isize) -> Result<(Tensor, Tensor)> {
    cumextreme(tensor, dim, false)
}

/// `log(sum(exp(x)))` accumulated along `dim`.
///
/// The running total of probabilities held as logarithms. Written as `cumsum`
/// of `exp` it underflows to zero after a few hundred steps and reports `-inf`
/// for everything after; combining in the log domain keeps every step
/// representable, which is why it exists as its own operation. Each prefix is
/// rounded once; see [`rounded_prefixes`].
pub fn logcumsumexp(tensor: &Tensor, dim: isize) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a variance, a norm and a log-sum-exp of integers are
    // all real numbers. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return crate::ops::util::narrowed(tensor, logcumsumexp(&widened, dim));
    }
    let dim = normalize_dim(dim, tensor.ndim())?;
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "logcumsumexp is only supported for floating point tensors",
        ));
    }
    let contiguous = tensor.contiguous()?;
    let (dim_size, inner) = slab_of(tensor, dim);
    let numel = tensor.numel();
    let device = tensor.device();
    let mut data = TensorData::zeros_on_device(numel, tensor.dtype(), device);
    let mismatch = || MinitensorError::internal_error("logcumsumexp: dtype does not match");
    if tensor.dtype() == DataType::Float32 {
        let source = contiguous.data().as_f32_slice().ok_or_else(mismatch)?;
        let out = data.as_f32_slice_mut().ok_or_else(mismatch)?;
        rounded_prefixes(out, &|at| f64::from(source[at]), dim_size, inner);
    } else {
        let source = contiguous.data().as_f64_slice().ok_or_else(mismatch)?;
        let out = data.as_f64_slice_mut().ok_or_else(mismatch)?;
        rounded_prefixes(out, &|at| source[at], dim_size, inner);
    }

    let mut result = Tensor::new(
        Arc::new(data),
        tensor.shape().clone(),
        tensor.dtype(),
        device,
        tensor.requires_grad(),
    );

    if result.requires_grad() {
        let grad_fn = Arc::new(LogcumsumexpBackward {
            input_id: tensor.id(),
            input: tensor.detach(),
            dim,
        });
        result = with_grad_fn(result, grad_fn)?;
    }
    Ok(result)
}

/// Each prefix's `log(Σ exp(x))` along the runs of slabs of `dim_size` rows of
/// `inner`, `read` giving the input by index: the float nearest the value its
/// [`Prefix`] carries, which is within about `2^-100` of the exact one.
///
/// Folding each value into the rounded running value, as `log_add_exp` does,
/// put a rounding into every step that every later one kept: of a thousand
/// float64 prefixes most were an ulp or more out, and of a hundred thousand
/// some a hundred, where the float32 nearest a prefix could be missed too.
fn rounded_prefixes<T: Output>(
    out: &mut [T],
    read: &(dyn Fn(usize) -> f64 + Sync),
    dim_size: usize,
    inner: usize,
) {
    let slab = dim_size * inner;
    if slab == 0 {
        return;
    }
    par_out_chunks(out, slab, &|start, chunk| {
        for (first, slab) in (start..).step_by(slab).zip(chunk.chunks_mut(slab)) {
            let read: &dyn Fn(usize) -> f64 = &|at| read(first + at);
            let emit: &mut dyn FnMut(usize, f64, f64) = &mut |at, value, rest| {
                slab[at] = T::settle(value, rest, 0.0).unwrap_or_else(|| T::narrow(value));
            };
            extended::fused!(
                prefixes(
                    dim_size: usize,
                    inner: usize,
                    read: &dyn Fn(usize) -> f64,
                    emit: &mut dyn FnMut(usize, f64, f64),
                ) -> ()
            )
        }
    });
}

/// A number far below the pairs it sits beside, `mantissa · 2^exponent` with
/// `|mantissa|` in `[1, 2)` or zero and the exponent kept apart however far
/// down it goes: the parts of a sum only a tie of it can show, kept to say
/// which way the sum lies from the halfway point.
#[derive(Clone, Copy)]
struct Faint {
    mantissa: f64,
    exponent: i64,
}

impl Faint {
    const ZERO: Self = Self {
        mantissa: 0.0,
        exponent: 0,
    };

    #[inline(never)]
    fn of(value: f64, exponent: i64) -> Self {
        let scaled = Scaled::of(Pair(value, 0.0), exponent);
        Self {
            mantissa: scaled.hi,
            exponent: scaled.exponent,
        }
    }

    /// The sum, the smaller brought to the larger's scale; one more than
    /// 2^1000 below the other cannot move its sign.
    #[inline(never)]
    fn plus(self, other: Self) -> Self {
        let (high, low) =
            if other.mantissa == 0.0 || (self.mantissa != 0.0 && self.exponent >= other.exponent) {
                (self, other)
            } else {
                (other, self)
            };
        let gap = high.exponent - low.exponent;
        if low.mantissa == 0.0 || gap > 1000 {
            return high;
        }
        Self::of(
            high.mantissa + low.mantissa * f64::from_bits(((1023 - gap) as u64) << 52),
            high.exponent,
        )
    }

    #[inline(never)]
    fn times(self, factor: f64, exponent: i64) -> Self {
        Self::of(self.mantissa * factor, self.exponent + exponent)
    }

    /// The value in places of `2^exponent`, or the least subnormal of its
    /// sign where that is too small to show.
    #[inline(never)]
    fn at(self, exponent: i64) -> f64 {
        let k = (self.exponent - exponent).clamp(-1100, 1000);
        let half = f64::from_bits(((1023 + k / 2) as u64) << 52);
        shrunk(
            shrunk(self.mantissa, half),
            f64::from_bits(((1023 + k - k / 2) as u64) << 52),
        )
    }
}

/// A prefix of a run as [`prefixes`] carries it: the largest value so far,
/// `top`, and the excess of `Σ exp(x - top)` over one, a pair of floats, so
/// that a total near one keeps its small part. `top` is `-inf` before any
/// value but `-inf` has come, and NaN or `+inf` once one has, which every
/// later prefix then is.
#[derive(Clone, Copy)]
struct Prefix {
    top: f64,
    excess: Pair,
    /// The terms below `exp(-708)` of the largest, which the pair cannot
    /// hold beside one: nothing to the prefix's logarithm, but in a slope
    /// they can break a tie.
    faint: Faint,
}

impl Prefix {
    const NONE: Self = Self {
        top: f64::NEG_INFINITY,
        excess: Pair(0.0, 0.0),
        faint: Faint::ZERO,
    };

    /// The prefix with `value` taken in: its term `exp(value - top)` of the
    /// exact difference, as a pair, or the total so far rescaled to a new
    /// largest. A term under `exp(-708)` of the largest, or a total so far
    /// that small beside a new one, is far below what the sum's logarithm
    /// holds.
    #[inline(always)]
    fn then<const FMA: bool>(mut self, value: f64) -> Self {
        if self.top.is_nan() || value.is_nan() {
            self.top = f64::NAN;
        } else if self.top == f64::INFINITY || value == f64::NEG_INFINITY {
        } else if value == f64::INFINITY || self.top == f64::NEG_INFINITY {
            self.top = value;
        } else {
            // The smaller of the two below the larger, whose own term is one.
            let larger = value > self.top;
            let (d, e) = if larger {
                two_sum(self.top, -value)
            } else {
                two_sum(value, -self.top)
            };
            if d < -708.0 {
                let far = Scaled::exp::<FMA>(Pair(d, e));
                let far = Faint::of(far.hi, far.exponent);
                self.faint = if larger {
                    // The whole total so far, far below the new largest.
                    let total = Faint::of(1.0 + self.excess.0, 0).plus(self.faint);
                    self.excess = Pair(0.0, 0.0);
                    total.times(far.mantissa, far.exponent)
                } else {
                    self.faint.plus(far)
                };
            } else {
                let term = extended::exp_pair::<FMA>(Pair(d, e));
                if larger {
                    self.excess = self.excess.add(Pair(1.0, 0.0)).mul::<FMA>(term);
                    self.faint = self.faint.times(term.0, 0);
                } else {
                    self.excess = self.excess.add(term);
                }
            }
            if larger {
                self.top = value;
            }
        }
        self
    }

    /// `top + log1p(excess)` as the float64 nearest it and what that left.
    #[inline(always)]
    fn value<const FMA: bool>(self) -> (f64, f64) {
        if !self.top.is_finite() {
            return (self.top, 0.0);
        }
        let l = extended::ln_1p::<FMA>(self.excess);
        let (sum, error) = two_sum(self.top, l.0);
        two_sum(sum, error + l.1)
    }
}

/// The prefixes of one slab's runs, `dim_size` long and `inner` side by side,
/// each given to `emit` with its index as a float64 and what it left.
fn prefixes<const FMA: bool>(
    dim_size: usize,
    inner: usize,
    read: &dyn Fn(usize) -> f64,
    emit: &mut dyn FnMut(usize, f64, f64),
) {
    let mut runs = vec![Prefix::NONE; inner];
    for step in 0..dim_size {
        for (lane, prefix) in runs.iter_mut().enumerate() {
            let at = step * inner + lane;
            *prefix = prefix.then::<FMA>(read(at));
            let (value, rest) = prefix.value::<FMA>();
            emit(at, value, rest);
        }
    }
}

/// Route each output's gradient back to the input position that supplied it.
pub(crate) fn cumextreme_backward(
    indices: &Tensor,
    grad_output: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    let positions = indices.contiguous()?;
    let seeds = grad_output.contiguous()?;
    let (dim_size, inner) = slab_of(indices, dim);
    let numel = indices.numel();
    let device = grad_output.device();
    let mut data = TensorData::zeros_on_device(numel, grad_output.dtype(), device);
    let source = positions
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("cummax backward: indices are not int64"))?;

    macro_rules! route {
        ($accessor:ident, $accessor_mut:ident) => {{
            let grad = seeds.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("cummax backward: dtype does not match")
            })?;
            let out = data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("cummax backward: dtype does not match")
            })?;
            let slab = dim_size * inner;
            if slab > 0 {
                // Serial within a slab: several prefixes can name the same input
                // position, and they accumulate there. Slabs are disjoint, so
                // the outer walk could be parallel -- it is not, because the
                // scatter is a handful of adds per element and the parallel
                // split would cost more than it saved.
                for slab_start in (0..numel).step_by(slab) {
                    for d in 0..dim_size {
                        for i in 0..inner {
                            let at = slab_start + d * inner + i;
                            let target = slab_start + source[at] as usize * inner + i;
                            out[target] += grad[at];
                        }
                    }
                }
            }
        }};
    }

    match grad_output.dtype() {
        DataType::Float32 => route!(as_f32_slice, as_f32_slice_mut),
        _ => route!(as_f64_slice, as_f64_slice_mut),
    }

    Ok(Tensor::new(
        Arc::new(data),
        indices.shape().clone(),
        grad_output.dtype(),
        device,
        false,
    ))
}

/// The gradient of [`logcumsumexp`]: `x[i]`'s slope `Σ_{k ≥ i} g[k] s_k(i)`,
/// for `g` the output's gradient and `s_k` the softmax of the prefix to `k`
/// -- the slope `logsumexp` gives each of a slice's entries -- the float
/// nearest it as the prefixes' pairs carry it; see [`run_slopes`].
///
/// It was `exp(x[i] + log(Σ_{k ≥ i} g[k] exp(-y[k])))`, the sums run as
/// logarithms over the rounded outputs, positive and negative `g` apart and
/// taken from each other at the end: every step's rounding went into the
/// exponent, float64 slopes were tens to thousands of ulps out, and where the
/// two parts cancelled a float32 slope thousands.
pub(crate) fn logcumsumexp_backward(
    input: &Tensor,
    grad_output: &Tensor,
    dim: usize,
) -> Result<Tensor> {
    // The runs last, each contiguous.
    let order: Vec<isize> = (0..input.ndim())
        .filter(|&axis| axis != dim)
        .chain([dim])
        .map(|axis| axis as isize)
        .collect();
    let x = input.detach().permute(order.clone())?.contiguous()?;
    let g = grad_output
        .detach()
        .astype(input.dtype())?
        .permute(order.clone())?
        .contiguous()?;
    let len = input.shape().dims()[dim];
    let chunk = len * (4096 / len.max(1)).max(1);
    let mismatch = || MinitensorError::internal_error("logcumsumexp backward: dtype");
    let data = if input.dtype() == DataType::Float32 {
        let (x, g) = (
            x.data().as_f32_slice().ok_or_else(mismatch)?,
            g.data().as_f32_slice().ok_or_else(mismatch)?,
        );
        let mut out = vec![0.0f32; x.len()];
        par_out_chunks(&mut out, chunk, &|start, runs| {
            slopes_into(
                &|at| f64::from(x[start + at]),
                &|at| f64::from(g[start + at]),
                len,
                runs,
            )
        });
        TensorData::from_vec_f32(out, input.device())
    } else {
        let (x, g) = (
            x.data().as_f64_slice().ok_or_else(mismatch)?,
            g.data().as_f64_slice().ok_or_else(mismatch)?,
        );
        let mut out = vec![0.0f64; x.len()];
        par_out_chunks(&mut out, chunk, &|start, runs| {
            slopes_into(&|at| x[start + at], &|at| g[start + at], len, runs)
        });
        TensorData::from_vec_f64(out, input.device())
    };
    let mut inverse = vec![0isize; order.len()];
    for (at, &axis) in order.iter().enumerate() {
        inverse[axis as usize] = at as isize;
    }
    Tensor::new(
        Arc::new(data),
        x.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    )
    .permute(inverse)?
    .contiguous()
}

/// The slopes of `runs`, contiguous runs of `len`, into `out`: each slope a
/// special value or a scaled pair from [`run_slopes`], rounded once.
fn slopes_into<T: Output>(
    x: &dyn Fn(usize) -> f64,
    g: &dyn Fn(usize) -> f64,
    len: usize,
    out: &mut [T],
) {
    let mut states = Vec::with_capacity(len);
    for (first, run) in (0..).step_by(len.max(1)).zip(out.chunks_mut(len.max(1))) {
        let x: &dyn Fn(usize) -> f64 = &|at| x(first + at);
        let g: &dyn Fn(usize) -> f64 = &|at| g(first + at);
        let states = &mut states;
        let emit: &mut dyn FnMut(usize, Scaled) = &mut |at, slope| {
            run[at] = nearest_scaled(slope);
        };
        // One build, without the fused multiply-add: the size of a second
        // is worth more than the fifth of the time it would save.
        run_slopes::<false>(len, x, g, states, emit)
    }
}

/// A value `(hi + lo) · 2^exponent + crumb`, `|hi|` in `[1, 2)` or both
/// zero: `crumb` what sums of the pair could not hold, found exactly and
/// kept beside it at its own scale, so that where the rest of a slope sums
/// exactly to a halfway point -- as shares of one and a half do -- the parts
/// far below it still say which way it lies.
#[derive(Clone, Copy)]
struct Scaled {
    hi: f64,
    lo: f64,
    exponent: i64,
    crumb: Faint,
}

impl Scaled {
    const ZERO: Self = Self {
        hi: 0.0,
        lo: 0.0,
        exponent: 0,
        crumb: Faint::ZERO,
    };

    /// `pair · 2^exponent`, normalized, its exponent kept within `±2^60`.
    #[inline(always)]
    fn of(mut pair: Pair, mut exponent: i64) -> Self {
        if pair.0 == 0.0 || !pair.0.is_finite() {
            return Self {
                hi: pair.0,
                exponent,
                ..Self::ZERO
            };
        }
        if pair.0.abs() < f64::MIN_POSITIVE {
            // A subnormal leading part, brought up into the normal range.
            pair = Pair(pair.0 * 2f64.powi(200), pair.1 * 2f64.powi(200));
            exponent -= 200;
        }
        let shift = ((pair.0.to_bits() >> 52) & 0x7ff) as i64 - 1023;
        let scale = f64::from_bits(((1023 - shift) as u64) << 52);
        Self {
            hi: pair.0 * scale,
            lo: pair.1 * scale,
            exponent: (exponent + shift).clamp(-(1 << 60), 1 << 60),
            crumb: Faint::ZERO,
        }
    }

    /// `exp(d)` for `d` at most zero, as a scaled pair: below `exp(-708)` the
    /// whole powers of two taken out first, `d - m log 2` formed as pairs.
    /// Past `-10^15` it is as small as anything here needs it to be.
    #[inline(always)]
    fn exp<const FMA: bool>(d: Pair) -> Self {
        let d = if d.0 < -1e15 { Pair(-1e15, 0.0) } else { d };
        let m = if d.0 >= -708.0 {
            0.0
        } else {
            (d.0 * extended::LOG2_E.0).round()
        };
        let r = d.add(Pair(m, 0.0).mul::<FMA>(extended::LN_2).neg());
        Self::of(extended::exp_pair::<FMA>(r), m as i64)
    }

    #[inline(always)]
    fn times<const FMA: bool>(self, other: Self) -> Self {
        Self {
            crumb: self.crumb.times(other.hi, other.exponent),
            ..Self::of(
                Pair(self.hi, self.lo).mul::<FMA>(Pair(other.hi, other.lo)),
                self.exponent + other.exponent,
            )
        }
    }

    /// The sum with a pair `term` not scaled and its own faint `tail`, the
    /// smaller brought to the larger's scale: the leading parts' sum and the
    /// trailing parts' exactly, the errors of adding what those leave into
    /// the crumb, and a part past the pair's places into it whole.
    #[inline(never)]
    fn plus(self, term: Pair, tail: Faint) -> Self {
        let term = Self::of(term, 0);
        let crumb = self.crumb.plus(tail);
        if term.hi == 0.0 || self.hi == 0.0 {
            let whole = if term.hi == 0.0 { self } else { term };
            return Self { crumb, ..whole };
        }
        let (high, low) = if self.exponent >= term.exponent {
            (self, term)
        } else {
            (term, self)
        };
        let gap = high.exponent - low.exponent;
        let mut crumb = crumb;
        let mut part = |value: f64| {
            let scaled = value * f64::from_bits(((1023 - gap.min(1000)) as u64) << 52);
            if gap > 1000 || (scaled.abs() < f64::MIN_POSITIVE && value != 0.0) {
                crumb = crumb.plus(Faint::of(value, low.exponent));
                0.0
            } else {
                scaled
            }
        };
        let (a, b) = (part(low.hi), part(low.lo));
        let (sum, carried) = two_sum(high.hi, a);
        let (rest, rest_error) = two_sum(high.lo, b);
        let (tail, tail_error) = two_sum(carried, rest);
        let (hi, lo) = two_sum(sum, tail);
        Self {
            crumb: crumb.plus(Faint::of(rest_error + tail_error, high.exponent)),
            ..Self::of(Pair(hi, lo), high.exponent)
        }
    }
}

/// `value · scale`, a power of two at most one, or the least subnormal of
/// its sign where that leaves nothing of a nonzero value.
#[inline(always)]
fn shrunk(value: f64, scale: f64) -> f64 {
    let scaled = value * scale;
    if scaled == 0.0 && value != 0.0 {
        f64::from_bits(1).copysign(value)
    } else {
        scaled
    }
}

/// The float nearest `slope`, ties to even: its pair need not be normalized,
/// and its crumb, smaller, breaks a tie the pair sits on exactly. Each slope is rounded on the
/// grid of its own last place, a subnormal's too, so that its parts are not
/// rounded twice on the way.
fn nearest_scaled<T: Output>(slope: Scaled) -> T {
    if slope.hi == 0.0 && slope.lo == 0.0 && slope.crumb.mantissa != 0.0 {
        // Nothing but the crumb, which is then the value, at its own scale.
        let crumb = slope.crumb;
        return nearest_scaled(Scaled::of(Pair(crumb.mantissa, 0.0), crumb.exponent));
    }
    let (hi, lo, exponent) = (slope.hi, slope.lo, slope.exponent);
    let crumb = slope.crumb.at(exponent);
    if lo == 0.0 && crumb == 0.0 && (hi == 0.0 || !hi.is_finite()) {
        // A signed zero, an infinity or a NaN as it stands.
        return T::narrow(hi);
    }
    let (lo, crumb) = two_sum(lo, crumb);
    let (hi, lo) = two_sum(hi, lo);
    if hi == 0.0 {
        // Parts that cancel exactly come to a positive zero.
        return T::narrow(0.0);
    }
    let lo = if crumb > 0.0 {
        lo.next_up()
    } else if crumb < 0.0 {
        lo.next_down()
    } else {
        lo
    };
    let scaled = Scaled::of(Pair(hi, lo), exponent);
    let (hi, lo) = two_sum(scaled.hi, scaled.lo);
    let scaled = Scaled::of(Pair(hi, lo), scaled.exponent);
    let (sign, m, l) = if scaled.hi < 0.0 {
        (-1.0, -scaled.hi, -scaled.lo)
    } else {
        (1.0, scaled.hi, scaled.lo)
    };
    let e = scaled.exponent;
    if e > i64::from(T::GREATEST) {
        return T::narrow(sign * f64::INFINITY);
    }
    if e < i64::from(T::LEAST) - 2 {
        return T::narrow(sign * 0.0);
    }
    // In places of the grid: `whole` exactly, below `2^digits`, and `rest`
    // under half a place of it.
    let grid = (e - i64::from(T::DIGITS - 1)).max(i64::from(T::LEAST));
    let up = f64::from_bits(((1023 + e - grid) as u64) << 52);
    let (whole, rest) = (m * up, l * up);
    let floor = whole.floor();
    let fraction = whole - floor;
    let odd = floor % 2.0 == 1.0;
    let count = if fraction == 0.0 {
        // A float64 grid: `rest` is at most half a place, and only exactly
        // half of one is a tie.
        if rest == 0.5 && odd {
            floor + 1.0
        } else if rest == -0.5 && odd {
            floor - 1.0
        } else {
            floor
        }
    } else if fraction > 0.5 || (fraction == 0.5 && (rest > 0.0 || (rest == 0.0 && odd))) {
        floor + 1.0
    } else {
        floor
    };
    let unit = if grid < -1022 {
        f64::from_bits(1 << (grid + 1074))
    } else {
        f64::from_bits(((grid + 1023) as u64) << 52)
    };
    T::narrow(sign * (count * unit))
}

/// The kind of the slope a prefix gives an entry finite, `-inf` and `+inf`
/// (see `exact_product::kind_of`): softmax's -- a NaN prefix NaN for each, an
/// infinite one a positive share to each `+inf` and nothing to the rest, and
/// a finite one a positive share to each finite entry; entries no prefix of
/// that kind holds are given nothing.
#[inline(always)]
fn slope_kinds(top: f64) -> [u8; 3] {
    if top.is_nan() {
        [NAN_KIND; 3]
    } else if top == f64::INFINITY {
        [0, 0, 2]
    } else if top == f64::NEG_INFINITY {
        [0; 3]
    } else {
        [2, 0, 0]
    }
}

/// The slopes of one run of `len`: the prefixes and their counts of `+inf`
/// from the front, then from the end back, for a finite entry
/// `exp(x[i] - top_i) R_i` with `R_i = Σ_{k ≥ i} g[k] exp(top_i - top_k) /
/// (1 + excess_k)` over the finite prefixes -- `R_i = w_i + exp(top_i -
/// top_{i + 1}) R_{i + 1}` -- carried as a scaled pair, and for a `+inf` one
/// its shares `Σ g[k] / count_k` as a pair. The kinds of the terms, gathered
/// for each kind of entry, settle every slope that holds a NaN, an infinity
/// or only zeros; the rest are given to `emit` as scaled pairs.
fn run_slopes<const FMA: bool>(
    len: usize,
    x: &dyn Fn(usize) -> f64,
    g: &dyn Fn(usize) -> f64,
    states: &mut Vec<(Prefix, u32)>,
    emit: &mut dyn FnMut(usize, Scaled),
) {
    states.clear();
    let (mut prefix, mut infinities) = (Prefix::NONE, 0u32);
    for at in 0..len {
        let value = x(at);
        prefix = prefix.then::<FMA>(value);
        infinities += u32::from(value == f64::INFINITY);
        states.push((prefix, infinities));
    }
    let mut kinds = [0u8; 3];
    let (mut sum, mut shares) = (Scaled::ZERO, Pair(0.0, 0.0));
    let mut later_top = f64::NAN;
    for at in (0..len).rev() {
        let (prefix, count) = states[at];
        let weight = g(at);
        let term = kind_of(weight);
        for (kind, slope) in kinds.iter_mut().zip(slope_kinds(prefix.top)) {
            *kind |= 1 << kind_times(term, slope);
        }
        if prefix.top.is_finite() {
            // `w = g / (1 + excess)`, the second part from the first's
            // remainder, and the later sum brought down to this top.
            let total = prefix.excess.add(Pair(1.0, 0.0));
            let q = weight / total.0;
            let remainder = Pair(weight, 0.0).add(Pair(q, 0.0).mul::<FMA>(total).neg());
            let (w, rest) = two_sum(q, remainder.0 / total.0);
            // What the faint terms take from `w`, to first order: past
            // anything a pair holds, so the rest is further still.
            let tail = prefix.faint.times(-weight / (total.0 * total.0), 0);
            if later_top.is_finite() && later_top != prefix.top {
                let (d, e) = two_sum(prefix.top, -later_top);
                sum = sum.times::<FMA>(Scaled::exp::<FMA>(Pair(d, e)));
            }
            sum = sum.plus(Pair(w, rest), tail);
            later_top = prefix.top;
        } else if prefix.top == f64::INFINITY && weight.is_finite() {
            let c = f64::from(count);
            let q = weight / c;
            shares = shares.add(Pair(q, (-q).mul_add(c, weight) / c));
        }
        let value = x(at);
        let entry = if value == f64::NEG_INFINITY {
            1
        } else if value == f64::INFINITY {
            2
        } else {
            0
        };
        let plain = |value: f64| Scaled {
            hi: value,
            ..Scaled::ZERO
        };
        if let Some(special) = sum_of_kinds(kinds[entry]) {
            emit(at, plain(special));
        } else if entry == 2 {
            emit(at, Scaled::of(shares, 0));
        } else if value == prefix.top {
            // The largest so far: its own factor is one, exactly.
            emit(at, sum);
        } else {
            let (d, e) = two_sum(value, -prefix.top);
            emit(
                at,
                if d < -3000.0 {
                    plain(if sum.hi < 0.0 { -0.0 } else { 0.0 })
                } else {
                    Scaled::exp::<FMA>(Pair(d, e)).times::<FMA>(sum)
                },
            );
        }
    }
}
