// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Padding, in the three modes that differ in what they put outside the edge.
//!
//! Constant padding can be spelled with `cat` and a tensor of zeros, awkwardly.
//! Reflect and replicate cannot be spelled at all -- they read the input back
//! at reflected or clamped coordinates, which is index arithmetic no
//! composition of the existing ops performs.
//!
//! All three are one mechanism: every output position maps to an input
//! position, or to nothing. Constant is the mode where "or to nothing" happens
//! and the fill value is used; the other two always land somewhere real, which
//! is also why they accumulate gradient many-to-one and constant does not.
//! Writing the map once means the three modes cannot disagree about anything
//! except the coordinate transform, which is the only thing they should differ
//! in.

use crate::{
    autograd::{PadBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::map::par_out_chunks,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// What to put outside the input's edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PadMode {
    /// A fixed value.
    Constant,
    /// The input mirrored at the edge, without repeating the edge element:
    /// `[a b c]` padded by 2 on the left is `[c b a b c]`.
    Reflect,
    /// The edge element repeated: `[a b c]` becomes `[a a a b c]`.
    Replicate,
}

impl PadMode {
    /// Parse the name the Python layer passes through.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "constant" => Ok(PadMode::Constant),
            "reflect" => Ok(PadMode::Reflect),
            "replicate" | "edge" => Ok(PadMode::Replicate),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown pad mode {other:?}; expected \"constant\", \"reflect\" or \"replicate\""
            ))),
        }
    }
}

/// How much is added before and after each axis, innermost axis first.
///
/// This is the order the flat `padding` argument arrives in, and it is the one
/// PyTorch uses: `(left, right)` for the last axis, then the one before it, and
/// so on. It reads backwards compared to a shape, which is exactly why the
/// conversion happens once here rather than at every use.
pub(crate) fn resolve_padding(ndim: usize, padding: &[usize]) -> Result<Vec<(usize, usize)>> {
    if !padding.len().is_multiple_of(2) {
        return Err(MinitensorError::invalid_argument(
            "pad expects an even number of padding values, one pair per axis",
        ));
    }
    let pairs = padding.len() / 2;
    if pairs > ndim {
        return Err(MinitensorError::invalid_argument(format!(
            "pad got {pairs} pairs of padding for a tensor with {ndim} dimensions"
        )));
    }
    // Axes not mentioned are unpadded. Index 0 of the result is axis 0 of the
    // tensor, so the incoming pairs are laid in from the back.
    let mut resolved = vec![(0usize, 0usize); ndim];
    for (pair, chunk) in padding.chunks_exact(2).enumerate() {
        resolved[ndim - 1 - pair] = (chunk[0], chunk[1]);
    }
    Ok(resolved)
}

/// Where output coordinate `out` on this axis reads from, or `None` when it
/// falls outside the input and the fill value applies.
///
/// `Reflect` mirrors without repeating the edge, so a run of `n` has period
/// `2 * (n - 1)`; a single-element axis has no period and is handled by the
/// validation below rather than by a special case here.
#[inline]
fn source_coord(out: usize, before: usize, extent: usize, mode: PadMode) -> Option<usize> {
    let shifted = out as isize - before as isize;
    if shifted >= 0 && (shifted as usize) < extent {
        return Some(shifted as usize);
    }
    match mode {
        PadMode::Constant => None,
        PadMode::Replicate => Some(if shifted < 0 { 0 } else { extent - 1 }),
        PadMode::Reflect => {
            let last = extent as isize - 1;
            let period = 2 * last;
            // Fold into `[0, period)`, then mirror the far half back.
            let mut folded = shifted.rem_euclid(period);
            if folded > last {
                folded = period - folded;
            }
            Some(folded as usize)
        }
    }
}

/// Validate the request and work out the output shape.
fn pad_layout(tensor: &Tensor, pads: &[(usize, usize)], mode: PadMode) -> Result<Vec<usize>> {
    let dims = tensor.shape().dims();
    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (&(before, after), &extent)) in pads.iter().zip(dims.iter()).enumerate() {
        if mode == PadMode::Reflect && (before > 0 || after > 0) {
            if extent < 2 {
                return Err(MinitensorError::invalid_argument(format!(
                    "reflect padding needs at least 2 elements on axis {axis}, which has {extent}"
                )));
            }
            // Beyond this the reflection would fold back over itself, and there
            // is no agreed answer for what it should produce.
            if before >= extent || after >= extent {
                return Err(MinitensorError::invalid_argument(format!(
                    "reflect padding on axis {axis} must be smaller than the axis ({before}, {after}) against {extent}"
                )));
            }
        }
        if mode == PadMode::Replicate && (before > 0 || after > 0) && extent == 0 {
            return Err(MinitensorError::invalid_argument(format!(
                "replicate padding has no edge to repeat on empty axis {axis}"
            )));
        }
        out_dims.push(extent + before + after);
    }
    Ok(out_dims)
}

/// Map every output position to its source, once, so the forward and the
/// backward walk the same correspondence rather than deriving it twice.
///
/// `None` means the position is outside the input; only `Constant` produces it.
pub(crate) fn pad_source_map(
    in_dims: &[usize],
    out_dims: &[usize],
    pads: &[(usize, usize)],
    mode: PadMode,
) -> Vec<Option<usize>> {
    let numel: usize = out_dims.iter().product();
    let mut map = vec![None; numel];
    if numel == 0 {
        return map;
    }

    // Row-major strides of the *input*, which is what a mapped coordinate
    // tuple has to be recombined against.
    let mut in_strides = vec![1usize; in_dims.len()];
    for axis in (0..in_dims.len().saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1] * in_dims[axis + 1];
    }

    for (linear, slot) in map.iter_mut().enumerate() {
        let mut rest = linear;
        let mut source = 0usize;
        let mut inside = true;
        // Right to left, the order row-major strides divide in.
        for axis in (0..out_dims.len()).rev() {
            let coord = rest % out_dims[axis];
            rest /= out_dims[axis];
            match source_coord(coord, pads[axis].0, in_dims[axis], mode) {
                Some(c) => source += c * in_strides[axis],
                None => {
                    inside = false;
                    break;
                }
            }
        }
        *slot = inside.then_some(source);
    }
    map
}

/// Pad `tensor`, adding `padding` before and after each axis.
///
/// `padding` is flat and innermost-axis-first: `[left, right]` pads the last
/// axis, `[left, right, top, bottom]` pads the last two. Axes it does not reach
/// are left alone.
pub fn pad(tensor: &Tensor, padding: &[usize], mode: PadMode, value: f64) -> Result<Tensor> {
    let pads = resolve_padding(tensor.ndim(), padding)?;
    let out_dims = pad_layout(tensor, &pads, mode)?;
    let out_shape = Shape::new(out_dims.clone());

    let in_dims = tensor.shape().dims().to_vec();
    let contiguous = tensor.contiguous()?;
    let map = pad_source_map(&in_dims, &out_dims, &pads, mode);

    let mut output_data =
        TensorData::zeros_on_device(out_shape.numel(), tensor.dtype(), tensor.device());

    macro_rules! fill {
        ($accessor:ident, $accessor_mut:ident, $ty:ty, $fill:expr) => {{
            let src = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("pad: dtype does not match the input slice")
            })?;
            let dst = output_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("pad: dtype does not match the output slice")
            })?;
            let filler: $ty = $fill;
            par_out_chunks(dst, crate::ops::map::PAR_CHUNK, &|start, chunk| {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    *slot = match map[start + offset] {
                        Some(source) => src[source],
                        None => filler,
                    };
                }
            });
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => fill!(as_f32_slice, as_f32_slice_mut, f32, value as f32),
        DataType::Float64 => fill!(as_f64_slice, as_f64_slice_mut, f64, value),
        DataType::Int32 => fill!(as_i32_slice, as_i32_slice_mut, i32, value as i32),
        DataType::Int64 => fill!(as_i64_slice, as_i64_slice_mut, i64, value as i64),
        DataType::Bool => fill!(as_bool_slice, as_bool_slice_mut, bool, value != 0.0),
    }

    let mut output = Tensor::new(
        Arc::new(output_data),
        out_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(PadBackward {
            input_shape: in_dims,
            map,
            input_id: tensor.id(),
            ids: [tensor.id()],
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}
