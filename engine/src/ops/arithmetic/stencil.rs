// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The finite differences `gradient` takes, each the float nearest the
//! exact stencil; see `exact_sum::ratio_of_products`.

use std::sync::Arc;

use super::exact_sum::{Element, Output, Product, difference_over, ratio_of_products};
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// Slices a task takes, about this many samples between them.
const TASK: usize = 1 << 14;

/// The finite differences of `values` along `axis`, second order inside and
/// `edge_order` at the ends, `step` apart or at `positions` -- one a sample
/// along the axis, of the values' float dtype.
///
/// Evenly spaced, the derivative inside is `(f[i+1] - f[i-1]) / 2h`, and at
/// the ends `(f[1] - f[0]) / h` or `(-3 f[0] + 4 f[1] - f[2]) / 2h` and their
/// mirrors; at positions whose gaps are not all one exact step it is the
/// second-order stencil of the two gaps either side. Each is the float
/// nearest the exact value. A NaN or an infinity among the samples or the
/// positions a stencil reads gives what its weights give in float64 -- the
/// even stencil never reads the sample it is centred on -- and `step` is
/// read at the values' width.
pub fn finite_differences(
    values: &Tensor,
    step: f64,
    positions: Option<&Tensor>,
    axis: usize,
    edge_order: usize,
) -> Result<Tensor> {
    let dtype = values.dtype();
    if !matches!(dtype, DataType::Float32 | DataType::Float64) || axis >= values.ndim() {
        return Err(MinitensorError::invalid_operation(format!(
            "finite differences take a float tensor and an axis of it, got {dtype:?} and axis \
             {axis} of {} dimensions",
            values.ndim()
        )));
    }
    let dims = values.shape().dims();
    let len = dims[axis];
    if !(1..=2).contains(&edge_order) || len < edge_order + 1 {
        return Err(MinitensorError::invalid_operation(format!(
            "finite differences of edge order {edge_order} need more than {edge_order} samples, \
             got {len}"
        )));
    }
    if let Some(positions) = positions
        && (positions.dtype() != dtype || positions.ndim() != 1 || positions.numel() != len)
    {
        return Err(MinitensorError::invalid_operation(format!(
            "finite differences take one position a sample, of the values' dtype, got {:?} of \
             {:?}",
            positions.dtype(),
            positions.shape().dims()
        )));
    }
    if positions.is_none() && step == 0.0 {
        return Err(MinitensorError::invalid_operation(
            "finite differences need a step that is not zero",
        ));
    }
    let values = values.contiguous()?;
    let positions = positions.map(Tensor::contiguous).transpose()?;
    let inner = dims[axis + 1..].iter().product::<usize>();
    let missing = || MinitensorError::internal_error("finite differences: a float slice");
    let mut data = TensorData::uninitialized_on_device(values.numel(), dtype, values.device());
    if dtype == DataType::Float32 {
        let f = values.data().as_f32_slice().ok_or_else(missing)?;
        let x = positions
            .as_ref()
            .map(|x| x.data().as_f32_slice().ok_or_else(missing))
            .transpose()?;
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        let grid = Grid::of(x, f64::from(step as f32));
        differences(f, &grid, edge_order, out, len, inner);
    } else {
        let f = values.data().as_f64_slice().ok_or_else(missing)?;
        let x = positions
            .as_ref()
            .map(|x| x.data().as_f64_slice().ok_or_else(missing))
            .transpose()?;
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        differences(f, &Grid::of(x, step), edge_order, out, len, inner);
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(dims.to_vec()),
        dtype,
        values.device(),
        false,
    ))
}

/// Where the samples lie: one exact step apart, or at positions.
enum Grid<'a, T> {
    Even(f64),
    Uneven(&'a [T]),
}

impl<'a, T: Element> Grid<'a, T> {
    /// Positions whose gaps are all one step exactly are that step: the even
    /// stencil is the uneven one's exact value there, and it does not read
    /// the sample it is centred on.
    fn of(positions: Option<&'a [T]>, step: f64) -> Self {
        let Some(x) = positions else {
            return Self::Even(step);
        };
        let gap = |i: usize| {
            let (gap, error) = super::rounded_once::two_sum(x[i + 1].wide(), -x[i].wide());
            (error == 0.0 && gap.is_finite()).then_some(gap)
        };
        match gap(0) {
            Some(first) if first != 0.0 && (1..x.len() - 1).all(|i| gap(i) == Some(first)) => {
                Self::Even(first)
            }
            _ => Self::Uneven(x),
        }
    }
}

/// [`finite_differences`] of `f` laid out `[outer, len, inner]` along the
/// middle axis into `out`, laid out the same: ranges of positions along the
/// axis to a task, so that one long run spreads across the pool, and the
/// runs side by side at a position taken together, along the rows they lie
/// in.
fn differences<T: Element + Output>(
    f: &[T],
    grid: &Grid<T>,
    edge_order: usize,
    out: &mut [T],
    len: usize,
    inner: usize,
) {
    if out.is_empty() {
        return;
    }
    let outer = out.len() / (len * inner);
    // Positions a task takes, about `TASK` samples between them.
    let span = (TASK / inner).clamp(1, len);
    let pieces = len.div_ceil(span);
    let parts = crate::parallel::Parts::of(out);
    #[cfg(target_arch = "x86_64")]
    let fused = crate::ops::simd::simd_capabilities().avx2_fma;
    let task = |index: usize| {
        let (o, piece) = (index / pieces, index % pieces);
        let positions = piece * span..((piece + 1) * span).min(len);
        let base = o * len * inner;
        let slab = &f[base..base + len * inner];
        // SAFETY: each task's positions of each slab have their own places
        // in `out`.
        let answers = unsafe {
            std::slice::from_raw_parts_mut(
                parts.start().add(base + positions.start * inner),
                positions.len() * inner,
            )
        };
        #[cfg(target_arch = "x86_64")]
        if fused {
            // SAFETY: fma was detected on this CPU.
            unsafe { rows_fused(slab, inner, grid, edge_order, positions, answers) };
            return;
        }
        rows_stencil::<T, false>(slab, inner, grid, edge_order, positions, answers);
    };
    let tasks = outer * pieces;
    if f.len() < crate::ops::map::PAR_THRESHOLD / 8 {
        (0..tasks).for_each(task);
    } else {
        crate::parallel::for_each_task(tasks, &task);
    }
}

/// [`rows_stencil`] compiled with fused multiply-adds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn rows_fused<T: Element + Output>(
    slab: &[T],
    inner: usize,
    grid: &Grid<T>,
    edge_order: usize,
    positions: std::ops::Range<usize>,
    out: &mut [T],
) {
    rows_stencil::<T, true>(slab, inner, grid, edge_order, positions, out);
}

/// The differences at `positions` of each of the `inner` runs of a
/// `[len, inner]` slab into `out`, laid out `[positions, inner]`.
#[inline(always)]
fn rows_stencil<T: Element + Output, const FMA: bool>(
    slab: &[T],
    inner: usize,
    grid: &Grid<T>,
    edge_order: usize,
    positions: std::ops::Range<usize>,
    out: &mut [T],
) {
    let first = positions.start;
    for i in positions {
        let row = &mut out[(i - first) * inner..(i - first + 1) * inner];
        for (j, slot) in row.iter_mut().enumerate() {
            let at = |k: usize| slab[k * inner + j].wide();
            *slot = point::<T, FMA>(slab.len() / inner, i, at, grid, edge_order);
        }
    }
}

/// The difference at position `i` of a run of `n` samples, `at` reading
/// them.
#[inline(always)]
fn point<T: Element + Output, const FMA: bool>(
    n: usize,
    i: usize,
    at: impl Fn(usize) -> f64,
    grid: &Grid<T>,
    edge_order: usize,
) -> T {
    match *grid {
        Grid::Even(h) => {
            if 0 < i && i < n - 1 {
                even_difference::<T, FMA>(at(i + 1), at(i - 1), 2.0 * h)
            } else if edge_order == 1 {
                let (a, b) = if i == 0 { (0, 1) } else { (n - 2, n - 1) };
                even_difference::<T, FMA>(at(b), at(a), h)
            } else if i == 0 {
                even_edge::<T, FMA>([at(0), at(1), at(2)], h, false)
            } else {
                even_edge::<T, FMA>([at(n - 1), at(n - 2), at(n - 3)], h, true)
            }
        }
        Grid::Uneven(x) => {
            let place = |k: usize| x[k].wide();
            if 0 < i && i < n - 1 {
                uneven_inside::<T, FMA>(
                    [at(i - 1), at(i), at(i + 1)],
                    [place(i - 1), place(i), place(i + 1)],
                )
            } else if edge_order == 1 {
                let (a, b) = if i == 0 { (0, 1) } else { (n - 2, n - 1) };
                uneven_slope::<T, FMA>([at(a), at(b)], [place(a), place(b)])
            } else if i == 0 {
                uneven_first::<T, FMA>([at(0), at(1), at(2)], [place(0), place(1), place(2)])
            } else {
                uneven_last::<T, FMA>(
                    [at(n - 3), at(n - 2), at(n - 1)],
                    [place(n - 3), place(n - 2), place(n - 1)],
                )
            }
        }
    }
}

/// A float as a pair.
fn single(value: f64) -> (f64, f64) {
    (value, 0.0)
}

/// `x - y` as a pair.
fn gap(x: f64, y: f64) -> (f64, f64) {
    (x, -y)
}

/// `(ahead - behind) / divisor`.
#[inline(always)]
fn even_difference<T: Output, const FMA: bool>(ahead: f64, behind: f64, divisor: f64) -> T {
    if !(ahead.is_finite() && behind.is_finite() && divisor.is_finite()) {
        return T::narrow((ahead - behind) / divisor);
    }
    difference_over::<T, FMA>(ahead, behind, divisor).unwrap_or_else(|| {
        ratio_of_products::<T, FMA>(
            &[Product::new(
                [gap(ahead, behind), single(1.0), single(1.0)],
                false,
            )],
            &[Product::new(
                [single(divisor), single(1.0), single(1.0)],
                false,
            )],
        )
    })
}

/// The second-order end of an even grid, `(-3 f[0] + 4 f[1] - f[2]) / 2h`
/// from the end inward, its sign turned for the last end.
#[inline(always)]
fn even_edge<T: Output, const FMA: bool>(f: [f64; 3], h: f64, last: bool) -> T {
    if !(f.iter().all(|value| value.is_finite()) && h.is_finite()) {
        let sign = if last { -1.0 } else { 1.0 };
        let total = (-1.5 * sign / h) * f[0] + (2.0 * sign / h) * f[1] + (-0.5 * sign / h) * f[2];
        return T::narrow(total);
    }
    let term = |value: f64, weight: f64, negative: bool| {
        Product::new(
            [single(value), single(weight), single(1.0)],
            negative ^ last,
        )
    };
    ratio_of_products::<T, FMA>(
        &[
            term(f[0], 3.0, true),
            term(f[1], 4.0, false),
            term(f[2], 1.0, true),
        ],
        &[Product::new([single(h), single(2.0), single(1.0)], false)],
    )
}

/// `(f[1] - f[0]) / (x[1] - x[0])`.
#[inline(always)]
fn uneven_slope<T: Output, const FMA: bool>(f: [f64; 2], x: [f64; 2]) -> T {
    if !(f.iter().chain(&x).all(|value| value.is_finite())) {
        return T::narrow((f[1] - f[0]) / (x[1] - x[0]));
    }
    ratio_of_products::<T, FMA>(
        &[Product::new(
            [gap(f[1], f[0]), single(1.0), single(1.0)],
            false,
        )],
        &[Product::new(
            [gap(x[1], x[0]), single(1.0), single(1.0)],
            false,
        )],
    )
}

/// The second-order stencil at `x[1]` of the gaps either side:
/// `(dx1² (f[2] - f[1]) + dx2² (f[1] - f[0])) / (dx1 dx2 (dx1 + dx2))`.
#[inline(always)]
fn uneven_inside<T: Output, const FMA: bool>(f: [f64; 3], x: [f64; 3]) -> T {
    if !(f.iter().chain(&x).all(|value| value.is_finite())) {
        let (behind, ahead) = (x[1] - x[0], x[2] - x[1]);
        let span = behind + ahead;
        let before = -ahead / (behind * span);
        let middle = (ahead - behind) / (behind * ahead);
        let after = behind / (ahead * span);
        return T::narrow(before * f[0] + middle * f[1] + after * f[2]);
    }
    if let Some(step) = equal_gaps(x) {
        // The middle weight is exactly zero: the even stencil's value.
        return even_difference::<T, FMA>(f[2], f[0], 2.0 * step);
    }
    let (behind, ahead, span) = (gap(x[1], x[0]), gap(x[2], x[1]), gap(x[2], x[0]));
    ratio_of_products::<T, FMA>(
        &[
            Product::new([behind, behind, gap(f[2], f[1])], false),
            Product::new([ahead, ahead, gap(f[1], f[0])], false),
        ],
        &[Product::new([behind, ahead, span], false)],
    )
}

/// The one gap between three positions, where both are that float exactly.
#[inline(always)]
fn equal_gaps(x: [f64; 3]) -> Option<f64> {
    let (first, first_error) = super::rounded_once::two_sum(x[1], -x[0]);
    let (second, second_error) = super::rounded_once::two_sum(x[2], -x[1]);
    (first_error == 0.0 && second_error == 0.0 && first == second && first != 0.0).then_some(first)
}

/// The second-order stencil at the first of three samples:
/// `(t² f[1] - (2 dx1 dx2 + dx2²) f[0] - dx1² f[2]) / (dx1 dx2 t)`, for the
/// gaps `dx1` and `dx2` and their sum `t`.
#[inline(always)]
fn uneven_first<T: Output, const FMA: bool>(f: [f64; 3], x: [f64; 3]) -> T {
    if !(f.iter().chain(&x).all(|value| value.is_finite())) {
        let (first, second) = (x[1] - x[0], x[2] - x[1]);
        let total = first + second;
        let weights = [
            -(2.0 * first + second) / (first * total),
            total / (first * second),
            -first / (second * total),
        ];
        return T::narrow(weights[0] * f[0] + weights[1] * f[1] + weights[2] * f[2]);
    }
    if let Some(step) = equal_gaps(x) {
        return even_edge::<T, FMA>(f, step, false);
    }
    let (first, second, total) = (gap(x[1], x[0]), gap(x[2], x[1]), gap(x[2], x[0]));
    let (f0, f1, f2) = (single(f[0]), single(f[1]), single(f[2]));
    ratio_of_products::<T, FMA>(
        &[
            Product::new([total, total, f1], false),
            Product::new([first, second, f0], true),
            Product::new([first, second, f0], true),
            Product::new([second, second, f0], true),
            Product::new([first, first, f2], true),
        ],
        &[Product::new([first, second, total], false)],
    )
}

/// The second-order stencil at the last of three samples:
/// `(dx2² f[0] - t² f[1] + (2 dx1 dx2 + dx1²) f[2]) / (dx1 dx2 t)`.
#[inline(always)]
fn uneven_last<T: Output, const FMA: bool>(f: [f64; 3], x: [f64; 3]) -> T {
    if !(f.iter().chain(&x).all(|value| value.is_finite())) {
        let (penultimate, last) = (x[1] - x[0], x[2] - x[1]);
        let total = penultimate + last;
        let weights = [
            last / (penultimate * total),
            -total / (penultimate * last),
            (2.0 * last + penultimate) / (last * total),
        ];
        return T::narrow(weights[0] * f[0] + weights[1] * f[1] + weights[2] * f[2]);
    }
    if let Some(step) = equal_gaps(x) {
        return even_edge::<T, FMA>([f[2], f[1], f[0]], step, true);
    }
    let (first, second, total) = (gap(x[1], x[0]), gap(x[2], x[1]), gap(x[2], x[0]));
    let (f0, f1, f2) = (single(f[0]), single(f[1]), single(f[2]));
    ratio_of_products::<T, FMA>(
        &[
            Product::new([second, second, f0], false),
            Product::new([total, total, f1], true),
            Product::new([first, second, f2], false),
            Product::new([first, second, f2], false),
            Product::new([first, first, f2], false),
        ],
        &[Product::new([first, second, total], false)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn run(f: Vec<f64>, step: f64, x: Option<Vec<f64>>, edge_order: usize) -> Vec<f64> {
        let positions = x.map(tensor);
        let got = finite_differences(&tensor(f), step, positions.as_ref(), 0, edge_order).unwrap();
        got.data().as_f64_slice().unwrap().to_vec()
    }

    #[test]
    fn even_and_uneven_stencils() {
        let f = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        assert_eq!(run(f.clone(), 1.0, None, 1), vec![1.0, 1.5, 2.5, 3.5, 4.0]);
        assert_eq!(run(f.clone(), 1.0, None, 2), vec![0.5, 1.5, 2.5, 3.5, 4.5]);
        // Positions one exact step apart are that step.
        let even = Some(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(run(f.clone(), 0.0, even, 2), run(f.clone(), 0.5, None, 2));
        // A quadratic's derivative, exactly, at uneven positions.
        let x = vec![0.0, 1.0, 3.0, 4.0, 7.0];
        let square: Vec<f64> = x.iter().map(|v| v * v).collect();
        assert_eq!(
            run(square, 0.0, Some(x.clone()), 2),
            vec![0.0, 2.0, 6.0, 8.0, 14.0]
        );
    }

    #[test]
    fn each_difference_rounds_once() {
        // (1 + 2^-52 - 1) / 3: the difference is exact, and the quotient
        // the float nearest a third of the last bit.
        let f = vec![1.0, 0.0, 1.0 + 2f64.powi(-52)];
        let got = run(f, 1.5, None, 1);
        assert_eq!(got[1], 2f64.powi(-52) / 3.0);
        // A missing sample stays where it was put on an even grid.
        let got = run(vec![1.0, f64::NAN, 3.0, 4.0], 1.0, None, 1);
        assert!(got[0].is_nan() && got[1] == 1.0 && got[2].is_nan() && got[3] == 1.0);
    }

    #[test]
    fn refuses_what_it_cannot_difference() {
        let f = tensor(vec![1.0, 2.0]);
        assert!(finite_differences(&f, 1.0, None, 0, 2).is_err());
        assert!(finite_differences(&f, 0.0, None, 0, 1).is_err());
        assert!(finite_differences(&f, 1.0, None, 1, 1).is_err());
        assert!(finite_differences(&f, 1.0, Some(&tensor(vec![0.0])), 0, 1).is_err());
    }
}
