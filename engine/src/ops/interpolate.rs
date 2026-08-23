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
/// `lower` and `upper` are the two source indices and `weight` is how much of
/// the upper one to take. Nearest sets them equal with zero weight, so the
/// kernel below has one path rather than two.
pub(crate) struct AxisMap {
    lower: Vec<usize>,
    upper: Vec<usize>,
    weight: Vec<f64>,
}

impl AxisMap {
    /// How many output positions this axis has.
    pub(crate) fn len(&self) -> usize {
        self.lower.len()
    }

    pub(crate) fn lower(&self, index: usize) -> usize {
        self.lower[index]
    }

    pub(crate) fn upper(&self, index: usize) -> usize {
        self.upper[index]
    }

    pub(crate) fn weight(&self, index: usize) -> f64 {
        self.weight[index]
    }
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
    let mut weight = vec![0f64; out_size];
    if in_size == 0 || out_size == 0 {
        return AxisMap {
            lower,
            upper,
            weight,
        };
    }
    let last = in_size - 1;

    for index in 0..out_size {
        let source = match mode {
            // Nearest takes the cell the output position falls in, which is the
            // floor of the uniform mapping and needs no convention: there is no
            // interpolation for the endpoints to be preserved by.
            InterpolateMode::Nearest => {
                let scaled = index as f64 * in_size as f64 / out_size as f64;
                lower[index] = (scaled as usize).min(last);
                upper[index] = lower[index];
                continue;
            }
            InterpolateMode::Linear if align_corners => {
                if out_size == 1 {
                    0.0
                } else {
                    index as f64 * last as f64 / (out_size - 1) as f64
                }
            }
            InterpolateMode::Linear => {
                let scaled = (index as f64 + 0.5) * in_size as f64 / out_size as f64 - 0.5;
                // A position left of the first sample has nothing to its left,
                // so it reads the first sample rather than extrapolating.
                scaled.max(0.0)
            }
        };
        let floor = source.floor();
        let base = (floor as usize).min(last);
        lower[index] = base;
        upper[index] = (base + 1).min(last);
        weight[index] = source - floor;
    }

    AxisMap {
        lower,
        upper,
        weight,
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

macro_rules! interpolate_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        /// Gather every output position from its (up to) four sources.
        ///
        /// Separable, so the two axes are combined here rather than stored
        /// together: the row weight multiplies a pair of column-interpolated
        /// values. Nearest falls out of the same expression with both indices
        /// equal and the weight zero, which is why it is not a second loop.
        fn $name(
            input: &[$ty],
            planes: usize,
            in_h: usize,
            in_w: usize,
            rows: &AxisMap,
            cols: &AxisMap,
        ) -> Vec<$ty> {
            let (out_h, out_w) = (rows.lower.len(), cols.lower.len());
            let plane_out = out_h * out_w;
            let plane_in = in_h * in_w;
            let mut values = vec![0 as $ty; planes * plane_out];
            if plane_out == 0 || plane_in == 0 {
                return values;
            }

            par_out_chunks(&mut values, plane_out, &|first, out_plane| {
                let base = (first / plane_out) * plane_in;
                for oh in 0..out_h {
                    let top = base + rows.lower[oh] * in_w;
                    let bottom = base + rows.upper[oh] * in_w;
                    let row_weight = rows.weight[oh] as $ty;
                    for ow in 0..out_w {
                        let (left, right) = (cols.lower[ow], cols.upper[ow]);
                        let column_weight = cols.weight[ow] as $ty;
                        let above = input[top + left]
                            + (input[top + right] - input[top + left]) * column_weight;
                        let below = input[bottom + left]
                            + (input[bottom + right] - input[bottom + left]) * column_weight;
                        out_plane[oh * out_w + ow] = above + (below - above) * row_weight;
                    }
                }
            });
            values
        }
    };
}

interpolate_kernel!(interpolate_f32, f32, as_f32_slice);
interpolate_kernel!(interpolate_f64, f64, as_f64_slice);

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
    let rows = axis_map(in_h, out_h, mode, align_corners);
    let cols = axis_map(in_w, out_w, mode, align_corners);

    let contiguous = input.contiguous()?;
    let data = match input.dtype() {
        DataType::Float32 => {
            let source = contiguous.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("interpolate: dtype does not match the input")
            })?;
            TensorData::from_vec_f32(
                interpolate_f32(source, planes, in_h, in_w, &rows, &cols),
                input.device(),
            )
        }
        _ => {
            let source = contiguous.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("interpolate: dtype does not match the input")
            })?;
            TensorData::from_vec_f64(
                interpolate_f64(source, planes, in_h, in_w, &rows, &cols),
                input.device(),
            )
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
