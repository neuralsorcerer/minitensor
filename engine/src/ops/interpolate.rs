// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Resampling a signal to a different size without learning anything.
//!
//! `conv_transpose2d` grows a feature map with a learned kernel and
//! `adaptive_avg_pool2d` shrinks one by averaging. Neither resamples: a U-Net's
//! decoder, a feature pyramid and every segmentation head need to take a map
//! back to an earlier resolution *without* parameters, so that a skip
//! connection lines up.
//!
//! Nearest-neighbour could almost be assembled -- an integer scale factor is a
//! repeat along each axis -- but a non-integer one cannot, and bilinear cannot
//! at all: it reads each output position from a weighted pair of neighbours at
//! a fractional coordinate, which is a gather no composition of the existing
//! ops performs.
//!
//! The whole operation is separable, and that is what the implementation is
//! built on. Each axis contributes a pair of source indices and one weight per
//! output position, so the map is `O(out_h + out_w)` numbers rather than four
//! index-weight pairs for every output element -- which for a `256x256` map
//! would have been 16MB of indices to read once.
//!
//! Being linear in the input, the backward is the transpose of the forward: the
//! same indices and the same weights, scattered instead of gathered. That is
//! asserted directly, as `<interpolate(x), y> == <x, backward(y)>`.

use crate::ops::arithmetic::exact_sum::{Element, Output, weighted_quotient};
use crate::ops::arithmetic::rounded_once::two_sum;
use crate::{
    autograd::{InterpolateBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::map::par_out_chunks,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// How an output position is read from the input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpolateMode {
    /// The single closest source. Blocky, exact on the samples it lands on, and
    /// the only mode with no arithmetic to get wrong.
    Nearest,
    /// A weighted average of the two neighbours on each axis -- linear for a
    /// 3-D signal, bilinear for a 4-D one. It is the same rule either way, so
    /// there is one name for it.
    Linear,
}

impl InterpolateMode {
    /// Parse the name the Python layer passes through. `bilinear` and `linear`
    /// are the same mode; which one a caller writes depends only on how many
    /// spatial axes they have, and the rank already says that.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "nearest" => Ok(InterpolateMode::Nearest),
            "linear" | "bilinear" => Ok(InterpolateMode::Linear),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown interpolate mode {other:?}; expected \"nearest\", \"linear\" or \"bilinear\""
            ))),
        }
    }
}

/// Where each output position along one axis reads from.
///
/// `lower` and `upper` are the two source indices, and `remainder` over
/// `denominator` is how much of the upper one to take: the weight exactly,
/// a whole number over the one whole number the axis shares. Nearest sets
/// the indices equal with no remainder, so the kernels have one path rather
/// than two.
pub(crate) struct AxisMap {
    lower: Vec<usize>,
    upper: Vec<usize>,
    remainder: Vec<u64>,
    denominator: u64,
}

impl AxisMap {
    /// How many output positions this axis has.
    pub(crate) fn len(&self) -> usize {
        self.lower.len()
    }

    /// The source indices of output position `index` and the whole weights
    /// of each, over [`AxisMap::denominator`].
    #[inline]
    pub(crate) fn taps(&self, index: usize) -> [(usize, u64); 2] {
        let upper = self.remainder[index];
        [
            (self.lower[index], self.denominator - upper),
            (self.upper[index], upper),
        ]
    }

    /// The whole number every weight of the axis is over.
    pub(crate) fn denominator(&self) -> u64 {
        self.denominator
    }

    /// The axis's sources and whole weights side by side, the weights as
    /// float64s, which hold them exactly.
    pub(crate) fn widened(&self) -> Widened {
        let (low, high) = self
            .remainder
            .iter()
            .map(|&upper| ((self.denominator - upper) as f64, upper as f64))
            .unzip();
        Widened {
            lower: self.lower.clone(),
            upper: self.upper.clone(),
            low,
            high,
        }
    }
}

/// [`AxisMap::widened`]: for each output position, the two sources and the
/// whole weight of each.
pub(crate) struct Widened {
    pub(crate) lower: Vec<usize>,
    pub(crate) upper: Vec<usize>,
    pub(crate) low: Vec<f64>,
    pub(crate) high: Vec<f64>,
}

/// The source cells of output `(oh, ow)` of a plane `in_w` wide, and the
/// whole coefficient of each over the product of the axes' denominators.
#[inline]
pub(crate) fn plane_taps(
    rows: &AxisMap,
    cols: &AxisMap,
    oh: usize,
    ow: usize,
    in_w: usize,
) -> [(usize, u64); 4] {
    let ([(top, above), (bottom, below)], [(left, before), (right, after)]) =
        (rows.taps(oh), cols.taps(ow));
    [
        (top * in_w + left, above * before),
        (top * in_w + right, above * after),
        (bottom * in_w + left, below * before),
        (bottom * in_w + right, below * after),
    ]
}

/// Build the map for one axis.
///
/// Two coordinate conventions, and the difference between them is the whole of
/// what `align_corners` means. With it set, the first and last output positions
/// sit exactly on the first and last input samples, and everything between is
/// spaced to match -- which preserves the endpoints and distorts the spacing.
/// Without it, output positions are the centres of equal cells covering the
/// input, which keeps the spacing uniform and lands the endpoints half a cell
/// inside. The second is the right default: it is the one that makes
/// resampling twice by two the same as resampling once by four.
pub(crate) fn axis_map(
    in_size: usize,
    out_size: usize,
    mode: InterpolateMode,
    align_corners: bool,
) -> AxisMap {
    let mut lower = vec![0usize; out_size];
    let mut upper = vec![0usize; out_size];
    let mut remainder = vec![0u64; out_size];
    if in_size == 0 || out_size == 0 {
        return AxisMap {
            lower,
            upper,
            remainder,
            denominator: 1,
        };
    }
    let last = in_size - 1;

    // Each source position is a fraction of whole numbers, so its whole part
    // is an exact quotient and its weight a remainder over the denominator,
    // kept as the two whole numbers. Taken from a position in floats, the
    // weight carried the position's rounding, an ulp of the index rather than
    // of the weight: a few thousand samples in, a weight was good to twelve
    // digits.
    let (in_wide, out_wide) = (in_size as u128, out_size as u128);
    let denominator = match mode {
        InterpolateMode::Nearest => 1,
        InterpolateMode::Linear if align_corners => out_wide.saturating_sub(1).max(1),
        InterpolateMode::Linear => 2 * out_wide,
    };
    for index in 0..out_size {
        let index_wide = index as u128;
        let (numerator, denominator) = match mode {
            // Nearest takes the cell the output position falls in, which is the
            // floor of the uniform mapping and needs no convention: there is no
            // interpolation for the endpoints to be preserved by.
            InterpolateMode::Nearest => {
                let cell = index_wide * in_wide / out_wide;
                lower[index] = (cell.min(last as u128)) as usize;
                upper[index] = lower[index];
                continue;
            }
            InterpolateMode::Linear if align_corners => {
                if out_size == 1 {
                    (0, 1)
                } else {
                    (index_wide * last as u128, out_wide - 1)
                }
            }
            // `(index + 0.5) * in / out - 0.5`, over `2 * out`. A position left
            // of the first sample has nothing to its left, so it reads the
            // first sample rather than extrapolating.
            InterpolateMode::Linear => (
                ((2 * index_wide + 1) * in_wide).saturating_sub(out_wide),
                2 * out_wide,
            ),
        };
        let whole = numerator / denominator;
        let base = whole.min(last as u128) as usize;
        lower[index] = base;
        upper[index] = (base + 1).min(last);
        remainder[index] = (numerator % denominator) as u64;
    }

    AxisMap {
        lower,
        upper,
        remainder,
        denominator: denominator as u64,
    }
}

/// The output extent for a scale factor.
///
/// Truncated rather than rounded, which is what every implementation does and
/// is worth stating: a factor of `0.5` on an odd extent loses the odd sample
/// rather than inventing a place for it.
fn scaled_extent(extent: usize, factor: f64) -> Result<usize> {
    // NaN is named alongside the range test rather than left to fall out of a
    // negation: it compares false against everything, so `!(factor > 0.0)`
    // would catch it by accident and read as if it did not mean to.
    if factor.is_nan() || !factor.is_finite() || factor <= 0.0 {
        return Err(MinitensorError::invalid_argument(
            "interpolate scale_factor must be finite and greater than zero",
        ));
    }
    Ok((extent as f64 * factor) as usize)
}

/// Work out the target spatial extents from whichever of the two was given.
pub(crate) fn resolve_output_size(
    spatial: &[usize],
    size: Option<&[usize]>,
    scale_factor: Option<&[f64]>,
) -> Result<Vec<usize>> {
    match (size, scale_factor) {
        (Some(_), Some(_)) => Err(MinitensorError::invalid_argument(
            "interpolate takes size or scale_factor, not both",
        )),
        (None, None) => Err(MinitensorError::invalid_argument(
            "interpolate needs either size or scale_factor",
        )),
        (Some(size), None) => {
            if size.len() != spatial.len() {
                return Err(MinitensorError::invalid_argument(format!(
                    "interpolate got {} sizes for {} spatial dimensions",
                    size.len(),
                    spatial.len()
                )));
            }
            Ok(size.to_vec())
        }
        (None, Some(factors)) => {
            if factors.len() != spatial.len() {
                return Err(MinitensorError::invalid_argument(format!(
                    "interpolate got {} scale factors for {} spatial dimensions",
                    factors.len(),
                    spatial.len()
                )));
            }
            spatial
                .iter()
                .zip(factors)
                .map(|(&extent, &factor)| scaled_extent(extent, factor))
                .collect()
        }
    }
}

/// How many outputs of a row are taken at a time: a stretch the stack holds.
const STRETCH: usize = 64;

/// Every output position of every plane read from its (up to) four sources:
/// the value nearest the exact weighted sum of them, rounded once.
///
/// Each weight is a whole number over its axis's denominator, so an output
/// times the product of the two denominators is a sum of whole coefficients
/// times values. The products are exact in float64 -- a float32's times a
/// coefficient under 2^29 as it stands, and any other with the error of its
/// product beside it -- and are added by two-sums, the sum divided and its
/// exact remainder found, so that the quotient comes with how far it is from
/// the exact one. Where nothing is left over, the quotient is the exact
/// value and narrowing it rounds once, a halfway point included; otherwise
/// the pair is settled where its bound places it, and an output it does not
/// place is taken exactly by [`weighted_quotient`]. Formed as two
/// interpolations along the width and one along the height in the element
/// type, with the weights narrowed to it first, an output took a rounding at
/// each step. Nearest falls out of the same expression, a single source of
/// the whole weight.
fn interpolated<T: Element + Output + Default>(
    input: &[T],
    planes: usize,
    in_h: usize,
    in_w: usize,
    rows: &AxisMap,
    cols: &AxisMap,
) -> Vec<T> {
    let (out_h, out_w) = (rows.len(), cols.len());
    let (plane_in, plane_out) = (in_h * in_w, out_h * out_w);
    let mut values = vec![T::default(); planes * plane_out];
    if plane_out == 0 || plane_in == 0 {
        return values;
    }
    let divisor = rows.denominator() * cols.denominator();
    if divisor == 1 {
        // Every output one source of the whole weight: nearest, or a
        // resampling onto the samples themselves. Each is that value, a
        // negative zero made a positive one as a sum of it would be.
        par_out_chunks(&mut values, plane_out, &|first, out_plane| {
            let plane = &input[(first / plane_out) * plane_in..][..plane_in];
            for (oh, row) in out_plane.chunks_mut(out_w).enumerate() {
                let source = &plane[rows.taps(oh)[0].0 * in_w..][..in_w];
                for (ow, value) in row.iter_mut().enumerate() {
                    *value = T::narrow(source[cols.taps(ow)[0].0].wide() + 0.0);
                }
            }
        });
        return values;
    }
    let (lines, columns) = (rows.widened(), cols.widened());
    par_out_chunks(&mut values, plane_out, &|first, out_plane| {
        let plane = &input[(first / plane_out) * plane_in..][..plane_in];
        let mut open = Vec::new();
        #[cfg(target_arch = "x86_64")]
        if crate::ops::simd::simd_capabilities().avx2_fma {
            // SAFETY: avx2 and fma were detected on this CPU.
            unsafe {
                interpolate_plane_fma(plane, out_plane, &lines, &columns, in_w, divisor, &mut open)
            };
        } else {
            interpolate_plane::<T, false>(
                plane, out_plane, &lines, &columns, in_w, divisor, &mut open,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        interpolate_plane::<T, false>(plane, out_plane, &lines, &columns, in_w, divisor, &mut open);
        for &slot in &open {
            let taps = plane_taps(rows, cols, slot / out_w, slot % out_w, in_w);
            out_plane[slot] =
                weighted_quotient(taps.iter().map(|&(cell, c)| (c, plane[cell])), divisor);
        }
    });
    values
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn interpolate_plane_fma<T: Element + Output>(
    plane: &[T],
    out: &mut [T],
    rows: &Widened,
    cols: &Widened,
    in_w: usize,
    divisor: u64,
    open: &mut Vec<usize>,
) {
    interpolate_plane::<T, true>(plane, out, rows, cols, in_w, divisor, open)
}

/// One plane of [`interpolated`], its outputs side by side a stretch of a
/// row at a time, and those the bound leaves open into `open`.
#[inline(always)]
fn interpolate_plane<T: Element + Output, const FMA: bool>(
    plane: &[T],
    out: &mut [T],
    rows: &Widened,
    cols: &Widened,
    in_w: usize,
    divisor: u64,
    open: &mut Vec<usize>,
) {
    // A float32 times a whole number under 2^29 is exact in float64.
    let wide = !(std::mem::size_of::<T>() == 4 && divisor < 1 << 29);
    let product_error = crate::ops::simd::product_error::<FMA>;
    let over = divisor as f64;
    // The tail and the bound need no more than a near reciprocal: the bound
    // takes what it misses by.
    let inverse = 1.0 / over;
    let out_w = cols.lower.len();
    let mut settled = [false; STRETCH];
    // A stretch's sources, brought side by side first so that the
    // arithmetic runs over them in vector lanes.
    let mut sources = [[0.0f64; STRETCH]; 4];
    for (oh, row) in out.chunks_mut(out_w).enumerate() {
        let top = &plane[rows.lower[oh] * in_w..][..in_w];
        let bottom = &plane[rows.upper[oh] * in_w..][..in_w];
        let (above, below) = (rows.low[oh], rows.high[oh]);
        for (stretch, values) in row.chunks_mut(STRETCH).enumerate() {
            let first = stretch * STRETCH;
            let len = values.len();
            for k in 0..len {
                let (left, right) = (cols.lower[first + k], cols.upper[first + k]);
                sources[0][k] = top[left].wide();
                sources[1][k] = top[right].wide();
                sources[2][k] = bottom[left].wide();
                sources[3][k] = bottom[right].wide();
            }
            let (befores, afters) = (
                &cols.low[first..first + len],
                &cols.high[first..first + len],
            );
            let mut every = true;
            for (k, value) in values.iter_mut().enumerate() {
                let (before, after) = (befores[k], afters[k]);
                let x = [sources[0][k], sources[1][k], sources[2][k], sources[3][k]];
                let c = [above * before, above * after, below * before, below * after];
                let p = [c[0] * x[0], c[1] * x[1], c[2] * x[2], c[3] * x[3]];
                let lost = if wide {
                    [
                        product_error(c[0], x[0], p[0]),
                        product_error(c[1], x[1], p[1]),
                        product_error(c[2], x[2], p[2]),
                        product_error(c[3], x[3], p[3]),
                    ]
                } else {
                    [0.0; 4]
                };
                let (sum, first_rounding) = two_sum(p[0], p[1]);
                let (sum, second_rounding) = two_sum(sum, p[2]);
                let (sum, third_rounding) = two_sum(sum, p[3]);
                let errors = (first_rounding + second_rounding + third_rounding)
                    + ((lost[0] + lost[1]) + (lost[2] + lost[3]));
                // The seven additions of the errors drop under 2^-49 of what
                // they add.
                let error_bound = (first_rounding.abs()
                    + second_rounding.abs()
                    + third_rounding.abs()
                    + (lost[0].abs() + lost[1].abs())
                    + (lost[2].abs() + lost[3].abs()))
                    * 2f64.powi(-49);
                let quotient = sum / over;
                let back = quotient * over;
                // `sum - quotient * divisor`, exactly: the first difference
                // is of neighbours and the second of what is representable.
                let remainder = (sum - back) - product_error(quotient, over, back);
                let tail = (remainder + errors) * inverse;
                let exact = (errors == 0.0) & (error_bound == 0.0) & (remainder == 0.0);
                // The products' errors and the remainder are exact away from
                // underflow and from the split's overflow; a float32 is
                // always there, and a float64 lane that is not is left open.
                let magnitude = sum.abs();
                let mut ordinary = (magnitude == 0.0)
                    | ((magnitude >= 2f64.powi(-900)) & (magnitude <= 2f64.powi(900)));
                if wide {
                    for k in 0..4 {
                        let size = p[k].abs();
                        ordinary &= (x[k].abs() <= 2f64.powi(900))
                            & ((size == 0.0) | (size >= 2f64.powi(-900)));
                    }
                }
                let (whole, rest) = two_sum(quotient, tail);
                let bound = tail.abs() * 2f64.powi(-49) + error_bound * inverse * 2.0;
                let (rounded, clear) = T::settle_lanes(whole, rest, bound);
                *value = if exact {
                    // An exact quotient, a zero a positive one.
                    T::narrow(quotient + 0.0)
                } else {
                    rounded
                };
                settled[k] = ordinary & ((exact & quotient.is_finite()) | clear);
                every &= settled[k];
            }
            if !every {
                let at = oh * out_w + first;
                open.extend((0..values.len()).filter(|&k| !settled[k]).map(|k| at + k));
            }
        }
    }
}

/// Resample a `[N, C, H, W]` or `[N, C, L]` signal to a different size.
///
/// Give exactly one of `size` (the target extents) and `scale_factor` (one
/// multiplier per spatial axis).
///
/// `align_corners` only means anything for [`InterpolateMode::Linear`]; see
/// [`axis_map`] for what it selects.
pub fn interpolate(
    input: &Tensor,
    size: Option<&[usize]>,
    scale_factor: Option<&[f64]>,
    mode: InterpolateMode,
    align_corners: bool,
) -> Result<Tensor> {
    let ndim = input.ndim();
    if ndim != 3 && ndim != 4 {
        return Err(MinitensorError::invalid_operation(
            "interpolate expects a 3D [N, C, L] or 4D [N, C, H, W] tensor",
        ));
    }
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "interpolate is implemented only for floating point tensors",
        ));
    }

    let dims = input.shape().dims().to_vec();
    let spatial = &dims[2..];
    let target = resolve_output_size(spatial, size, scale_factor)?;
    if spatial.contains(&0) && target.iter().any(|&e| e > 0) {
        return Err(MinitensorError::invalid_argument(
            "interpolate cannot resample an empty axis into a non-empty one",
        ));
    }

    // A 3-D signal is a 4-D one with a singleton height, so there is one kernel
    // and one backward rather than two to keep in step -- the same arrangement
    // `conv1d` and the 1-D poolers use.
    let (in_h, in_w) = if ndim == 4 {
        (spatial[0], spatial[1])
    } else {
        (1, spatial[0])
    };
    let (out_h, out_w) = if ndim == 4 {
        (target[0], target[1])
    } else {
        (1, target[0])
    };

    let planes = dims[0] * dims[1];
    // The output is the input's planes times the requested spatial size, which
    // the caller chooses freely: interpolating a 2x2 image to 10**6 by 10**6
    // asked for 4 TB and aborted rather than refusing.
    TensorData::ensure_allocatable(
        planes
            .checked_mul(out_h)
            .and_then(|n| n.checked_mul(out_w))
            .unwrap_or(usize::MAX),
        input.dtype(),
    )?;
    // The index maps are built at the output extent whatever the plane count,
    // three eight-byte entries a position: with no planes the check above
    // passes for any size, and `Upsample(2**62)` on an empty batch overflowed
    // a capacity building them.
    TensorData::ensure_allocatable(
        out_h.saturating_add(out_w).saturating_mul(3),
        DataType::Float64,
    )?;
    let rows = axis_map(in_h, out_h, mode, align_corners);
    let cols = axis_map(in_w, out_w, mode, align_corners);

    let contiguous = input.contiguous()?;
    let mismatched =
        || MinitensorError::internal_error("interpolate: dtype does not match the input");
    let data = match input.dtype() {
        DataType::Float32 => {
            let source = contiguous.data().as_f32_slice().ok_or_else(mismatched)?;
            let values = interpolated(source, planes, in_h, in_w, &rows, &cols);
            TensorData::from_vec_f32(values, input.device())
        }
        _ => {
            let source = contiguous.data().as_f64_slice().ok_or_else(mismatched)?;
            let values = interpolated(source, planes, in_h, in_w, &rows, &cols);
            TensorData::from_vec_f64(values, input.device())
        }
    };

    let mut out_dims = dims[..2].to_vec();
    out_dims.extend_from_slice(&target);
    let mut output = Tensor::new(
        Arc::new(data),
        Shape::new(out_dims),
        input.dtype(),
        input.device(),
        input.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(InterpolateBackward {
            input_id: input.id(),
            input_shape: dims,
            output_size: target,
            mode,
            align_corners,
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}
