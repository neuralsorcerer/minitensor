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

/// The correspondence between an output position and its source, walked rather
/// than written down.
///
/// It used to be written down: a `Vec<Option<usize>>` with one entry per output
/// element, built in one serial pass and then kept alive in the graph for the
/// backward to read. On four million float32 that was 64MB of `Option<usize>`
/// -- sixteen times the tensor -- and 68ms against NumPy's 3.8 for a job that
/// is mostly a copy.
///
/// The walk is over *rows* instead: everything but the last axis is decomposed
/// once per row, and the last axis is three runs -- the margin before, the
/// input itself, the margin after. The middle run is a `copy_from_slice`, which
/// is what padding actually is, and the divisions that dominated now happen
/// once per row rather than once per element.
pub struct PadPlan {
    in_dims: Vec<usize>,
    out_dims: Vec<usize>,
    pads: Vec<(usize, usize)>,
    mode: PadMode,
    /// Row-major strides of the *input*, which is what a mapped coordinate
    /// tuple has to be recombined against.
    in_strides: Vec<usize>,
    out_last: usize,
    in_last: usize,
    before_last: usize,
}

impl PadPlan {
    pub fn new(
        in_dims: Vec<usize>,
        out_dims: Vec<usize>,
        pads: Vec<(usize, usize)>,
        mode: PadMode,
    ) -> Self {
        let ndim = out_dims.len();
        let mut in_strides = vec![1usize; in_dims.len()];
        for axis in (0..in_dims.len().saturating_sub(1)).rev() {
            in_strides[axis] = in_strides[axis + 1] * in_dims[axis + 1];
        }
        // A scalar has no last axis to run along, so it is treated as one row
        // of one element and every loop below degenerates correctly.
        let (out_last, in_last, before_last) = if ndim == 0 {
            (1, 1, 0)
        } else {
            (out_dims[ndim - 1], in_dims[ndim - 1], pads[ndim - 1].0)
        };
        Self {
            in_dims,
            out_dims,
            pads,
            mode,
            in_strides,
            out_last,
            in_last,
            before_last,
        }
    }

    /// Where output row `row` begins in the input, or `None` when the row lies
    /// outside it entirely -- which only constant padding produces.
    #[inline]
    fn source_row(&self, row: usize) -> Option<usize> {
        let mut rest = row;
        let mut base = 0usize;
        // Right to left, the order row-major strides divide in.
        for axis in (0..self.out_dims.len().saturating_sub(1)).rev() {
            let coord = rest % self.out_dims[axis];
            rest /= self.out_dims[axis];
            let source = source_coord(coord, self.pads[axis].0, self.in_dims[axis], self.mode)?;
            base += source * self.in_strides[axis];
        }
        Some(base)
    }

    /// Where output column `col` of a row reads from, or `None` for the fill.
    #[inline]
    fn source_column(&self, col: usize) -> Option<usize> {
        source_coord(col, self.before_last, self.in_last, self.mode)
    }

    /// Write one output row: the margin before, the input's own run, the margin
    /// after.
    #[inline]
    fn write_row<T: Copy>(&self, out: &mut [T], input: &[T], row: usize, filler: T) {
        let Some(base) = self.source_row(row) else {
            out.fill(filler);
            return;
        };
        let interior = &input[base..base + self.in_last];
        let after = self.before_last + self.in_last;
        out[self.before_last..after].copy_from_slice(interior);
        for (col, slot) in out[..self.before_last].iter_mut().enumerate() {
            *slot = match self.source_column(col) {
                Some(c) => interior[c],
                None => filler,
            };
        }
        for (offset, slot) in out[after..].iter_mut().enumerate() {
            *slot = match self.source_column(after + offset) {
                Some(c) => interior[c],
                None => filler,
            };
        }
    }

    /// Add one output row's gradient back onto the input positions it read.
    ///
    /// Many-to-one for reflect and replicate -- an edge element is read by
    /// several output positions -- which is why this accumulates and why the
    /// walk over rows stays sequential.
    #[inline]
    fn accumulate_row<T>(&self, out: &[T], input: &mut [T], row: usize)
    where
        T: Copy + std::ops::AddAssign,
    {
        let Some(base) = self.source_row(row) else {
            return;
        };
        let after = self.before_last + self.in_last;
        for (slot, value) in input[base..base + self.in_last]
            .iter_mut()
            .zip(&out[self.before_last..after])
        {
            *slot += *value;
        }
        for (col, value) in out[..self.before_last].iter().enumerate() {
            if let Some(c) = self.source_column(col) {
                input[base + c] += *value;
            }
        }
        for (offset, value) in out[after..].iter().enumerate() {
            if let Some(c) = self.source_column(after + offset) {
                input[base + c] += *value;
            }
        }
    }

    /// Fill a padded output from `input`, a band of rows at a time.
    pub fn scatter<T: Copy + Send + Sync>(&self, out: &mut [T], input: &[T], filler: T) {
        if self.out_last == 0 || out.is_empty() {
            return;
        }
        // Whole rows per task, so no chunk ever straddles one and the
        // per-row decomposition happens once.
        let rows_per_task = (crate::ops::map::PAR_CHUNK / self.out_last).max(1);
        par_out_chunks(out, rows_per_task * self.out_last, &|start, chunk| {
            let first = start / self.out_last;
            for (offset, row) in chunk.chunks_mut(self.out_last).enumerate() {
                self.write_row(row, input, first + offset, filler);
            }
        });
    }

    /// Add a padded gradient back onto the input's shape.
    pub fn gather<T>(&self, out: &[T], input: &mut [T])
    where
        T: Copy + std::ops::AddAssign,
    {
        if self.out_last == 0 {
            return;
        }
        for (row, chunk) in out.chunks(self.out_last).enumerate() {
            self.accumulate_row(chunk, input, row);
        }
    }
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
    let plan = PadPlan::new(in_dims.clone(), out_dims, pads, mode);

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
            plan.scatter(dst, src, $fill);
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
            plan,
            input_id: tensor.id(),
            ids: [tensor.id()],
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}
