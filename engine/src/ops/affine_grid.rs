// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The sampling grid an affine map describes, each entry rounded once.
//!
//! A grid point's coordinate along an axis of `L` samples is a whole number
//! over a whole one: `(2j + 1 - L) / L`, or with the corners aligned
//! `(2j + 1 - L) / (L - 1)` -- nothing over one for a single sample. An entry
//! of the grid is `Σ θ_k · c_k + θ_r`, which over the product `Q` of the
//! point's denominators is `Σ θ_k · w_k / Q` for whole weights `w_k`; each is
//! the float nearest that exact value, as `weighted_quotient` gives it. So is
//! each entry of the gradient in `θ`: `Σ_p g_p · c_k(p)`, a whole-weighted
//! sum over one denominator.
//!
//! An infinity or a NaN in `θ`, or in the gradient, gives what float64
//! products give it: a zero coordinate times an infinity is NaN.

use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{Element, Output, weighted_quotient},
    ops::map::par_out_chunks,
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// Grid points a task takes.
const TASK: usize = 1 << 12;

/// The coordinate of position `j` along an axis of `length` samples, a whole
/// numerator over a whole denominator.
fn coordinate(j: usize, length: usize, align_corners: bool) -> (i64, u64) {
    let numerator = 2 * j as i64 + 1 - length as i64;
    match (align_corners, length) {
        (true, 1) => (0, 1),
        (true, _) => (numerator, length as u64 - 1),
        (false, _) => (numerator, length as u64),
    }
}

/// The checks both directions make: a float `tensor` of `shape`, and two or
/// three positive `spatial` sizes whose denominators multiply out within
/// a float64's whole numbers.
fn checked(
    tensor: &Tensor,
    shape: &[usize],
    spatial: &[usize],
    align_corners: bool,
) -> Result<u64> {
    let rank = spatial.len();
    let float = matches!(tensor.dtype(), DataType::Float32 | DataType::Float64);
    let common = spatial.iter().try_fold(1u64, |product, &length| {
        product.checked_mul(coordinate(0, length, align_corners).1)
    });
    match common {
        Some(common)
            if float
                && (2..=3).contains(&rank)
                && spatial.iter().all(|&length| length > 0)
                && tensor.shape().dims() == shape
                && common < 1 << 52 =>
        {
            Ok(common)
        }
        _ => Err(MinitensorError::invalid_operation(format!(
            "affine_grid takes a float tensor of shape {shape:?} over two or three positive \
             sizes, got {:?} of shape {:?} over {spatial:?}",
            tensor.dtype(),
            tensor.shape().dims()
        ))),
    }
}

/// A tensor of `dtype` and `shape` filled by `fill`, given the float32 or
/// float64 slices of `input` and of the new data.
fn filled(
    input: &Tensor,
    shape: Vec<usize>,
    fill32: impl FnOnce(&[f32], &mut [f32]),
    fill64: impl FnOnce(&[f64], &mut [f64]),
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let count = shape.iter().product();
    let missing = || MinitensorError::internal_error("affine_grid: a float slice");
    let mut data = TensorData::uninitialized_on_device(count, input.dtype(), input.device());
    if input.dtype() == DataType::Float32 {
        let values = input.data().as_f32_slice().ok_or_else(missing)?;
        fill32(values, data.as_f32_slice_mut().ok_or_else(missing)?);
    } else {
        let values = input.data().as_f64_slice().ok_or_else(missing)?;
        fill64(values, data.as_f64_slice_mut().ok_or_else(missing)?);
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(shape),
        input.dtype(),
        input.device(),
        false,
    ))
}

/// The sampling grid of each affine map in `theta`, `[n, r, r + 1]`, over an
/// output of the `r` sizes `spatial`: `[n, *spatial, r]`, entry `i` of a
/// point `Σ θ[i, k] · c_k + θ[i, r]` for its coordinates in `x, y (, z)`
/// order -- the reverse of the axes they index -- each the float nearest
/// its exact value. No gradient: `affine_grid` gives it.
pub fn affine_grid(theta: &Tensor, spatial: &[usize], align_corners: bool) -> Result<Tensor> {
    let rank = spatial.len();
    let batch = theta.shape().dims().first().copied().unwrap_or(0);
    let common = checked(theta, &[batch, rank, rank + 1], spatial, align_corners)?;
    let mut shape = vec![batch];
    shape.extend_from_slice(spatial);
    shape.push(rank);
    let setting = (spatial, align_corners, common);
    filled(
        theta,
        shape,
        |theta, out| grid(theta, out, setting),
        |theta, out| grid(theta, out, setting),
    )
}

/// The weights of a point's coordinates over `common`, `x` first, and
/// `common` itself last for the constant.
fn weights(point: usize, (spatial, align_corners, common): (&[usize], bool, u64)) -> [i64; 4] {
    let mut weights = [0i64; 4];
    let mut rest = point;
    for (k, &length) in spatial.iter().rev().enumerate() {
        let (numerator, denominator) = coordinate(rest % length, length, align_corners);
        weights[k] = numerator * (common / denominator) as i64;
        rest /= length;
    }
    weights[spatial.len()] = common as i64;
    weights
}

/// [`affine_grid`] of `theta` into `out`, a block of grid points to a task.
fn grid<T: Element + Output>(theta: &[T], out: &mut [T], setting: (&[usize], bool, u64)) {
    let rank = setting.0.len();
    let points: usize = setting.0.iter().product();
    par_out_chunks(out, rank * TASK, &|start, part| {
        let (mut terms, mut whole) = (Vec::new(), Vec::new());
        let first = start / rank;
        for (offset, entries) in part.chunks_mut(rank).enumerate() {
            let flat = first + offset;
            let weights = weights(flat % points, setting);
            let maps = &theta[flat / points * rank * (rank + 1)..];
            for (i, entry) in entries.iter_mut().enumerate() {
                let row = &maps[i * (rank + 1)..(i + 1) * (rank + 1)];
                terms.clear();
                terms.extend(row.iter().copied().zip(weights));
                *entry = combination(&terms, setting.2, &mut whole);
            }
        }
    });
}

/// The value nearest `Σ w · x / divisor` over `terms`, each a value `x` and
/// a whole weight `w`; where a value is not finite, what float64 products
/// make of it. `whole` is room for the terms as `weighted_quotient` takes
/// them.
#[inline(never)]
fn combination<T: Element + Output>(
    terms: &[(T, i64)],
    divisor: u64,
    whole: &mut Vec<(u64, T)>,
) -> T {
    if terms.iter().any(|(x, _)| !x.wide().is_finite()) {
        return unbounded(terms);
    }
    whole.clear();
    whole.extend(terms.iter().filter(|&&(_, w)| w != 0).map(|&(x, w)| {
        let x = if w < 0 { T::narrow(-x.wide()) } else { x };
        (w.unsigned_abs(), x)
    }));
    weighted_quotient(whole.iter().copied(), divisor)
}

/// A sum of whole-weighted values among which an infinity or a NaN lies, as
/// float64 products make it: NaN for a NaN, a zero weight on an infinity or
/// infinite products of both signs, and otherwise the infinity.
#[cold]
fn unbounded<T: Output>(terms: &[(T, i64)]) -> T {
    let (mut up, mut down, mut nan) = (false, false, false);
    for &(x, w) in terms {
        let x = x.widen();
        if !x.is_finite() {
            let product = x * w as f64;
            nan |= product.is_nan();
            up |= product == f64::INFINITY;
            down |= product == f64::NEG_INFINITY;
        }
    }
    T::narrow(if nan || (up && down) {
        f64::NAN
    } else if up {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    })
}

/// The gradient of [`affine_grid`] in `theta` from `grad`, `[n, *spatial,
/// r]`: `[n, r, r + 1]`, entry `[b, i, k]` the float nearest
/// `Σ_p grad[b, p, i] · c_k(p)` over every point `p` -- and `[b, i, r]`, for
/// the constant, the float nearest `Σ_p grad[b, p, i]`.
pub fn affine_grid_backward(
    grad: &Tensor,
    spatial: &[usize],
    align_corners: bool,
) -> Result<Tensor> {
    let rank = spatial.len();
    let batch = grad.shape().dims().first().copied().unwrap_or(0);
    let mut shape = vec![batch];
    shape.extend_from_slice(spatial);
    shape.push(rank);
    checked(grad, &shape, spatial, align_corners)?;
    let setting = (spatial, align_corners);
    filled(
        grad,
        vec![batch, rank, rank + 1],
        |grad, out| grid_gradient(grad, out, setting),
        |grad, out| grid_gradient(grad, out, setting),
    )
}

/// [`affine_grid_backward`] of `grad` into `out`, an entry to a task.
fn grid_gradient<T: Element + Output>(
    grad: &[T],
    out: &mut [T],
    (spatial, align_corners): (&[usize], bool),
) {
    let rank = spatial.len();
    let points: usize = spatial.iter().product();
    par_out_chunks(out, 1, &|entry, place| {
        let (b, i, k) = (
            entry / (rank * (rank + 1)),
            entry / (rank + 1) % rank,
            entry % (rank + 1),
        );
        // The coordinate `k` reads at each point, over its denominator: one
        // over one for the constant.
        let (axis, length) = (
            rank.saturating_sub(k + 1),
            spatial[rank.saturating_sub(k + 1)],
        );
        let stride: usize = spatial[axis + 1..].iter().product();
        let read = |p: usize| match k == rank {
            true => (1, 1),
            false => coordinate(p / stride % length, length, align_corners),
        };
        let terms: Vec<(T, i64)> = (0..points)
            .map(|p| (grad[(b * points + p) * rank + i], read(p).0))
            .collect();
        place[0] = combination(&terms, read(0).1, &mut Vec::new());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_are_whole_numbers_over_the_length() {
        assert_eq!(coordinate(0, 4, false), (-3, 4));
        assert_eq!(coordinate(3, 4, false), (3, 4));
        assert_eq!(coordinate(0, 4, true), (-3, 3));
        assert_eq!(coordinate(2, 5, true), (0, 4));
        assert_eq!(coordinate(0, 1, true), (0, 1));
        assert_eq!(coordinate(0, 1, false), (0, 1));
    }

    #[test]
    fn a_weighted_sum_with_an_infinity_is_what_products_make() {
        let of = |terms: &[(f64, i64)], divisor| combination(terms, divisor, &mut Vec::new());
        assert!(of(&[(f64::INFINITY, 0), (1.0, 2)], 3).is_nan());
        assert!(of(&[(f64::INFINITY, 1), (f64::INFINITY, -1)], 3).is_nan());
        assert!(of(&[(f64::NAN, 1)], 3).is_nan());
        assert_eq!(of(&[(f64::INFINITY, -2), (1e308, 5)], 7), f64::NEG_INFINITY);
        // Whole weights and a finite total: rounded once.
        assert_eq!(of(&[(1.0, 1), (0.0, 5)], 3), 1.0 / 3.0);
        // `0.3 - 3 · 0.1` of the floats is `-2^-55`, where rounding the
        // product first leaves `-2^-54`.
        assert_eq!(of(&[(0.1, -3), (0.3, 1)], 1), -(2f64.powi(-55)));
    }
}
