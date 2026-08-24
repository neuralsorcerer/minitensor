// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Reading an image at coordinates rather than at indices.
//!
//! `grid_sample` takes an input and a field of sampling coordinates, and
//! returns the input read at those coordinates. Where `interpolate` resamples
//! onto a regular grid it computes for you, this takes the grid as an argument
//! -- and, crucially, differentiates with respect to it. That is what makes a
//! spatial transformer, an optical-flow warp or a deformable convolution
//! trainable: the network learns *where* to look, not just what to do with what
//! it found.
//!
//! It does not compose. `index_select` and `gather` read at integer indices and
//! have no derivative in them; the whole content here is that a fractional
//! coordinate reads a weighted blend of its neighbours, and that the blend's
//! weights are themselves functions of the coordinate.
//!
//! ## Coordinates
//!
//! The grid is normalised: `-1` is one edge of the input and `+1` the other,
//! whatever the input's size, so a grid can be used against inputs of different
//! resolutions. `align_corners` settles what those two values name -- the
//! centres of the corner pixels, or the outer edges of them. Neither is more
//! correct; they differ by half a pixel, and a model trained under one is wrong
//! under the other, so it is an argument rather than a decision made here.
//!
//! The grid's last axis is in `x, y` order for a 4-D input and `x, y, z` for a
//! 5-D one -- the *reverse* of the spatial axes it indexes, which are `H, W`
//! and `D, H, W`. That is the convention every framework uses and it is a
//! reliable source of silently transposed outputs, so it has its own test.
//!
//! ## Off the edge
//!
//! A coordinate outside `[-1, 1]` has to mean something. `zeros` reads nothing
//! there, `border` holds the edge value, and `reflection` folds back inside. The
//! last two are applied to the coordinate before any neighbour is chosen, so
//! they change *where* the sample comes from; `zeros` instead drops the
//! individual neighbours that fall outside, which is why a coordinate just
//! inside the edge still blends with nothing rather than with the edge.
//!
//! ## The derivative in the coordinate
//!
//! Folding and clamping are not smooth, and the derivative has to say so: a
//! coordinate held against an edge by `border` moves the output not at all, and
//! one folded back by `reflection` moves it the other way. [`locate`] therefore
//! returns the source position *and* how fast it moves, and the coordinate
//! gradient is that rate times the derivative of the blend. Getting the rate
//! wrong is invisible in the forward pass and turns training into a slow drift
//! in the wrong direction, so the sign is tested directly.

use crate::{
    autograd::{GridSampleBackward, with_grad_fn},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

/// How the neighbours around a coordinate are combined.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleMode {
    /// A weighted blend of the `2^d` surrounding samples. The only mode with a
    /// derivative in the coordinate, and so the only one a spatial transformer
    /// can train through.
    Bilinear,
    /// The single closest sample. Exact where it lands, and flat everywhere --
    /// its coordinate gradient is zero, not merely small.
    Nearest,
}

/// What lies outside the input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Padding {
    /// Nothing. Neighbours outside the input contribute zero.
    Zeros,
    /// The nearest edge value, held indefinitely.
    Border,
    /// The input mirrored about its edges, repeatedly.
    Reflection,
}

impl SampleMode {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "bilinear" => Ok(SampleMode::Bilinear),
            "nearest" => Ok(SampleMode::Nearest),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown grid_sample mode {other:?}; expected \"bilinear\" or \"nearest\""
            ))),
        }
    }
}

impl Padding {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "zeros" => Ok(Padding::Zeros),
            "border" => Ok(Padding::Border),
            "reflection" => Ok(Padding::Reflection),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown grid_sample padding mode {other:?}; expected \"zeros\", \"border\" or \"reflection\""
            ))),
        }
    }
}

/// Hold `value` inside `[0, size - 1]`, reporting whether it moved.
///
/// The rate is zero once clamped: pushing a coordinate further past the edge
/// does not move the sample, so it must not move the gradient either.
fn clamp(value: f64, size: usize) -> (f64, f64) {
    let top = size as f64 - 1.0;
    if value <= 0.0 {
        (0.0, 0.0)
    } else if value >= top {
        (top, 0.0)
    } else {
        (value, 1.0)
    }
}

/// Fold `value` back into `[low, high]` by reflecting off each end in turn.
///
/// Every fold reverses the direction of travel, which is the whole reason this
/// reports a rate rather than just a position.
fn reflect(value: f64, low: f64, high: f64) -> (f64, f64) {
    let span = high - low;
    if span <= 0.0 {
        return (low, 0.0);
    }
    let shifted = value - low;
    let (distance, rate) = if shifted < 0.0 {
        (-shifted, -1.0)
    } else {
        (shifted, 1.0)
    };
    let folds = (distance / span).floor();
    let extra = distance - folds * span;
    if (folds as i64) % 2 == 0 {
        (extra + low, rate)
    } else {
        (span - extra + low, -rate)
    }
}

/// Where a normalised coordinate reads from, and how fast that moves.
///
/// The rate is the derivative of the source position with respect to the
/// coordinate: the scale of the normalisation, negated by every reflection and
/// zeroed by a clamp. It is what carries the gradient back to the grid.
fn locate(coord: f64, size: usize, padding: Padding, align_corners: bool) -> (f64, f64) {
    // `align_corners` decides whether -1 and 1 name the centres of the corner
    // samples or the outer edges of them -- half a pixel apart, and a different
    // scale as a result.
    let (position, mut rate) = if align_corners {
        (
            (coord + 1.0) * 0.5 * (size as f64 - 1.0),
            (size as f64 - 1.0) * 0.5,
        )
    } else {
        (((coord + 1.0) * size as f64 - 1.0) * 0.5, size as f64 * 0.5)
    };

    match padding {
        // Out-of-range neighbours are dropped later, one at a time, so the
        // coordinate itself is left exactly where it fell.
        Padding::Zeros => (position, rate),
        Padding::Border => {
            let (held, moved) = clamp(position, size);
            (held, rate * moved)
        }
        Padding::Reflection => {
            // Reflecting about the same two points the normalisation used, so
            // that -1 and 1 stay fixed under the fold.
            let (low, high) = if align_corners {
                (0.0, size as f64 - 1.0)
            } else {
                (-0.5, size as f64 - 0.5)
            };
            let (folded, turned) = reflect(position, low, high);
            rate *= turned;
            // A fold can still leave a coordinate half a pixel outside when the
            // corners are not aligned, so the clamp is not redundant.
            let (held, moved) = clamp(folded, size);
            (held, rate * moved)
        }
    }
}

/// The taps of one sample: flat offsets into a spatial plane, their weights,
/// and how each weight moves with each coordinate.
///
/// There are two or three axes -- an image or a volume -- and everything below
/// is written once against that count, because a trilinear blend is a bilinear
/// blend with one more axis and nothing else changes. Three axes is the most
/// there can be, so the arrays are fixed at their widest and `count` says how
/// much of them is in use.
struct Taps {
    /// `1 << axes` corners; `None` for one that falls outside the input.
    offset: [Option<usize>; 8],
    weight: [f64; 8],
    /// `slope[axis][corner]`: d(weight)/d(source position along `axis`).
    slope: [[f64; 8]; 3],
    count: usize,
}

impl Taps {
    /// The corners around `source`, with the blend weights and their slopes.
    ///
    /// `source[a]` is a position along spatial axis `a`, already folded or
    /// clamped by [`locate`]; `sizes[a]` is that axis's extent and `strides[a]`
    /// its stride within one plane.
    fn bilinear(
        source: &[f64; 3],
        sizes: &[usize; 3],
        strides: &[usize; 3],
        axes: usize,
        slopes: bool,
    ) -> Self {
        let mut base = [0i64; 3];
        let mut high = [0.0f64; 3];
        for axis in 0..axes {
            let floor = source[axis].floor();
            base[axis] = floor as i64;
            high[axis] = source[axis] - floor;
        }

        let count = 1usize << axes;
        let mut taps = Taps {
            offset: [None; 8],
            weight: [0.0; 8],
            slope: [[0.0; 8]; 3],
            count,
        };
        for corner in 0..count {
            let mut offset = 0usize;
            let mut inside = true;
            let mut weight = 1.0f64;
            for axis in 0..axes {
                let upper = corner >> axis & 1 == 1;
                let index = base[axis] + upper as i64;
                if index < 0 || index >= sizes[axis] as i64 {
                    inside = false;
                }
                // Clamped only so the arithmetic is defined: an index that
                // needed clamping is out of bounds, and `inside` already says
                // the tap contributes nothing.
                let held = index.clamp(0, (sizes[axis] as i64 - 1).max(0));
                offset += held as usize * strides[axis];
                weight *= if upper { high[axis] } else { 1.0 - high[axis] };
            }
            taps.offset[corner] = inside.then_some(offset);
            taps.weight[corner] = weight;
            // d(weight)/d(source[axis]) is the same product with this axis's
            // factor replaced by its own derivative, which is +1 for the upper
            // neighbour and -1 for the lower one. Only the backward pass wants
            // it, and it is `axes << axes` more products per output position,
            // so the forward pass does not pay for it.
            if !slopes {
                continue;
            }
            for axis in 0..axes {
                let upper = corner >> axis & 1 == 1;
                let mut slope = if upper { 1.0 } else { -1.0 };
                for other in 0..axes {
                    if other != axis {
                        let up = corner >> other & 1 == 1;
                        slope *= if up { high[other] } else { 1.0 - high[other] };
                    }
                }
                taps.slope[axis][corner] = slope;
            }
        }
        taps
    }

    /// The single closest corner, with weight one and no slope at all.
    ///
    /// Not a special case of the blend: rounding is flat between samples, so
    /// the coordinate gradient here is exactly zero rather than a small number.
    fn nearest(source: &[f64; 3], sizes: &[usize; 3], strides: &[usize; 3], axes: usize) -> Self {
        let mut offset = 0usize;
        let mut inside = true;
        for axis in 0..axes {
            // Round half to even, matching every other implementation of this.
            let index = source[axis].round_ties_even() as i64;
            if index < 0 || index >= sizes[axis] as i64 {
                inside = false;
            }
            offset += index.clamp(0, (sizes[axis] as i64 - 1).max(0)) as usize * strides[axis];
        }
        let mut taps = Taps {
            offset: [None; 8],
            weight: [0.0; 8],
            slope: [[0.0; 8]; 3],
            count: 1,
        };
        taps.offset[0] = inside.then_some(offset);
        taps.weight[0] = 1.0;
        taps
    }
}

/// One batch element's worth of geometry, shared by the forward and backward.
struct Field {
    axes: usize,
    sizes: [usize; 3],
    strides: [usize; 3],
    plane: usize,
    channels: usize,
    positions: usize,
    mode: SampleMode,
    padding: Padding,
    align_corners: bool,
}

impl Field {
    /// The taps and coordinate rates for one output position.
    ///
    /// `slopes` asks for the weight derivatives too, which only the backward
    /// pass has any use for.
    fn at(&self, coords: &[f64], slopes: bool) -> (Taps, [f64; 3]) {
        let mut source = [0.0f64; 3];
        let mut rate = [0.0f64; 3];
        for axis in 0..self.axes {
            // The grid names its axes in reverse: `x` first, indexing the last
            // spatial axis.
            let (position, moved) = locate(
                coords[self.axes - 1 - axis],
                self.sizes[axis],
                self.padding,
                self.align_corners,
            );
            source[axis] = position;
            rate[axis] = moved;
        }
        let taps = match self.mode {
            SampleMode::Bilinear => {
                Taps::bilinear(&source, &self.sizes, &self.strides, self.axes, slopes)
            }
            SampleMode::Nearest => Taps::nearest(&source, &self.sizes, &self.strides, self.axes),
        };
        (taps, rate)
    }

    /// Read every channel at every output position of one batch element.
    fn read(&self, input: &[f64], grid: &[f64], out: &mut [f64]) {
        for position in 0..self.positions {
            let coords = &grid[position * self.axes..(position + 1) * self.axes];
            let (taps, _) = self.at(coords, false);
            for channel in 0..self.channels {
                let plane = &input[channel * self.plane..(channel + 1) * self.plane];
                let mut total = 0.0;
                for corner in 0..taps.count {
                    if let Some(offset) = taps.offset[corner] {
                        total += taps.weight[corner] * plane[offset];
                    }
                }
                out[channel * self.positions + position] = total;
            }
        }
    }

    /// Push one batch element's output gradient back to the input and the grid.
    fn write(
        &self,
        input: &[f64],
        grid: &[f64],
        upstream: &[f64],
        into_input: Option<&mut [f64]>,
        into_grid: Option<&mut [f64]>,
    ) {
        let mut into_input = into_input;
        let mut into_grid = into_grid;
        for position in 0..self.positions {
            let coords = &grid[position * self.axes..(position + 1) * self.axes];
            let (taps, rate) = self.at(coords, into_grid.is_some());
            let mut moved = [0.0f64; 3];
            for channel in 0..self.channels {
                let seed = upstream[channel * self.positions + position];
                if seed == 0.0 {
                    continue;
                }
                let base = channel * self.plane;
                for corner in 0..taps.count {
                    let Some(offset) = taps.offset[corner] else {
                        continue;
                    };
                    if let Some(target) = into_input.as_deref_mut() {
                        target[base + offset] += seed * taps.weight[corner];
                    }
                    if into_grid.is_some() {
                        let value = input[base + offset];
                        for axis in 0..self.axes {
                            moved[axis] += seed * value * taps.slope[axis][corner];
                        }
                    }
                }
            }
            if let Some(target) = into_grid.as_deref_mut() {
                for axis in 0..self.axes {
                    // Back through the coordinate map, which is where a clamped
                    // or reflected coordinate loses or reverses its gradient.
                    target[position * self.axes + self.axes - 1 - axis] = moved[axis] * rate[axis];
                }
            }
        }
    }
}

/// Widen a tensor's data to `f64`, whatever it arrived as.
fn widen(tensor: &Tensor) -> Result<Vec<f64>> {
    let contiguous = tensor.contiguous()?;
    match tensor.dtype() {
        DataType::Float32 => Ok(contiguous
            .data()
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("grid_sample: dtype mismatch"))?
            .iter()
            .map(|&value| value as f64)
            .collect()),
        DataType::Float64 => Ok(contiguous
            .data()
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("grid_sample: dtype mismatch"))?
            .to_vec()),
        _ => Err(MinitensorError::invalid_operation(
            "grid_sample: input and grid must be floating point tensors",
        )),
    }
}

/// Narrow `values` back into a tensor of `dtype`.
fn narrow(
    values: &[f64],
    shape: Shape,
    dtype: DataType,
    device: crate::device::Device,
) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(values.len(), dtype, device);
    match dtype {
        DataType::Float32 => {
            let out = data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("grid_sample: dtype mismatch"))?;
            for (slot, item) in out.iter_mut().zip(values) {
                *slot = *item as f32;
            }
        }
        _ => {
            let out = data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("grid_sample: dtype mismatch"))?;
            out.copy_from_slice(values);
        }
    }
    Ok(Tensor::new(Arc::new(data), shape, dtype, device, false))
}

/// Everything the forward and the backward both need to agree on.
struct Layout {
    batch: usize,
    channels: usize,
    axes: usize,
    sizes: [usize; 3],
    strides: [usize; 3],
    plane: usize,
    positions: usize,
    out_dims: Vec<usize>,
}

fn plan(input: &Tensor, grid: &Tensor) -> Result<Layout> {
    let rank = input.ndim();
    if rank != 4 && rank != 5 {
        return Err(MinitensorError::invalid_operation(
            "grid_sample: input must be (batch, channels, height, width) or (batch, channels, depth, height, width)",
        ));
    }
    if grid.ndim() != rank {
        return Err(MinitensorError::invalid_operation(format!(
            "grid_sample: a {rank}-dimensional input needs a {rank}-dimensional grid"
        )));
    }
    let axes = rank - 2;
    let dims = input.shape().dims();
    let grid_dims = grid.shape().dims();
    if grid_dims[0] != dims[0] {
        return Err(MinitensorError::invalid_operation(
            "grid_sample: the grid and the input must agree on the batch size",
        ));
    }
    if grid_dims[rank - 1] != axes {
        return Err(MinitensorError::invalid_operation(format!(
            "grid_sample: the grid's last axis must hold {axes} coordinates, not {}",
            grid_dims[rank - 1]
        )));
    }

    let mut sizes = [1usize; 3];
    sizes[..axes].copy_from_slice(&dims[2..2 + axes]);
    let mut strides = [1usize; 3];
    for axis in (0..axes.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * sizes[axis + 1];
    }
    let plane: usize = sizes[..axes].iter().product();
    let positions: usize = grid_dims[1..rank - 1].iter().product();

    let mut out_dims = vec![dims[0], dims[1]];
    out_dims.extend_from_slice(&grid_dims[1..rank - 1]);
    Ok(Layout {
        batch: dims[0],
        channels: dims[1],
        axes,
        sizes,
        strides,
        plane,
        positions,
        out_dims,
    })
}

impl Layout {
    /// The geometry one batch element is read with.
    fn field(&self, mode: SampleMode, padding: Padding, align_corners: bool) -> Field {
        Field {
            axes: self.axes,
            sizes: self.sizes,
            strides: self.strides,
            plane: self.plane,
            channels: self.channels,
            positions: self.positions,
            mode,
            padding,
            align_corners,
        }
    }
}

/// Read `input` at the coordinates in `grid`.
///
/// `input` is `(batch, channels, ...spatial)` and `grid` is
/// `(batch, ...output spatial, axes)` holding coordinates in `[-1, 1]`. The
/// result is `(batch, channels, ...output spatial)`.
pub fn grid_sample(
    input: &Tensor,
    grid: &Tensor,
    mode: SampleMode,
    padding: Padding,
    align_corners: bool,
) -> Result<Tensor> {
    if input.dtype() != grid.dtype() {
        return Err(MinitensorError::invalid_operation(
            "grid_sample: the input and the grid must share a dtype",
        ));
    }
    let layout = plan(input, grid)?;
    let values = widen(input)?;
    let coords = widen(grid)?;

    let per_input = layout.channels * layout.plane;
    let per_grid = layout.positions * layout.axes;
    let per_out = layout.channels * layout.positions;
    let mut out = vec![0.0f64; layout.batch * per_out];

    let field = layout.field(mode, padding, align_corners);
    // `chunks_mut` will not take a zero width, and a batch with no channels or
    // an empty grid has nothing to read anyway.
    if per_out > 0 {
        out.par_chunks_mut(per_out)
            .enumerate()
            .for_each(|(index, into)| {
                field.read(
                    &values[index * per_input..(index + 1) * per_input],
                    &coords[index * per_grid..(index + 1) * per_grid],
                    into,
                );
            });
    }

    // Nothing above ran a tensor operation, so there is no primitive graph to
    // suppress and no guard to open -- the result is built element by element.
    let needs_grad =
        crate::autograd::is_grad_enabled() && (input.requires_grad() || grid.requires_grad());
    let sampled = narrow(
        &out,
        Shape::new(layout.out_dims.clone()),
        input.dtype(),
        input.device(),
    )?;
    if !needs_grad {
        return Ok(sampled);
    }

    let grad_fn = Arc::new(GridSampleBackward {
        input_ids: [input.id(), grid.id()],
        input_requires_grad: [input.requires_grad(), grid.requires_grad()],
        input: input.detach(),
        grid: grid.detach(),
        mode,
        padding,
        align_corners,
    });
    with_grad_fn(sampled.requires_grad_(true), grad_fn)
}

/// One batch element's two gradient slices, either of which may be unwanted.
type Slots<'a> = (Option<&'a mut [f64]>, Option<&'a mut [f64]>);

/// The gradients of [`grid_sample`] with respect to its input and its grid.
///
/// Recomputed rather than stored. The taps are the same shape as the forward
/// pass and cost the same to find again, which is cheaper than carrying a
/// `2^axes`-wide weight table for every output position through to the backward
/// pass -- and far cheaper when only one of the two gradients is wanted.
pub(crate) fn grid_sample_backward(
    input: &Tensor,
    grid: &Tensor,
    upstream: &Tensor,
    wanted: [bool; 2],
    mode: SampleMode,
    padding: Padding,
    align_corners: bool,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let layout = plan(input, grid)?;
    let values = widen(input)?;
    let coords = widen(grid)?;
    let seeds = widen(upstream)?;

    let per_input = layout.channels * layout.plane;
    let per_grid = layout.positions * layout.axes;
    let per_out = layout.channels * layout.positions;

    let mut input_grad = vec![0.0f64; if wanted[0] { values.len() } else { 0 }];
    let mut grid_grad = vec![0.0f64; if wanted[1] { coords.len() } else { 0 }];

    // Different batch elements touch disjoint planes of both gradients, so the
    // scatter is safe in parallel exactly at this granularity and no finer.
    let field = layout.field(mode, padding, align_corners);
    {
        let input_chunks: Vec<&mut [f64]> = if wanted[0] && per_input > 0 {
            input_grad.chunks_mut(per_input).collect()
        } else {
            Vec::new()
        };
        let grid_chunks: Vec<&mut [f64]> = if wanted[1] && per_grid > 0 {
            grid_grad.chunks_mut(per_grid).collect()
        } else {
            Vec::new()
        };
        let mut input_iter = input_chunks.into_iter();
        let mut grid_iter = grid_chunks.into_iter();
        let slots: Vec<Slots> = (0..layout.batch)
            .map(|_| (input_iter.next(), grid_iter.next()))
            .collect();
        slots
            .into_par_iter()
            .enumerate()
            .for_each(|(index, (into_input, into_grid))| {
                field.write(
                    &values[index * per_input..(index + 1) * per_input],
                    &coords[index * per_grid..(index + 1) * per_grid],
                    &seeds[index * per_out..(index + 1) * per_out],
                    into_input,
                    into_grid,
                );
            });
    }

    let for_input = wanted[0]
        .then(|| {
            narrow(
                &input_grad,
                input.shape().clone(),
                input.dtype(),
                input.device(),
            )
        })
        .transpose()?;
    let for_grid = wanted[1]
        .then(|| {
            narrow(
                &grid_grad,
                grid.shape().clone(),
                grid.dtype(),
                grid.device(),
            )
        })
        .transpose()?;
    Ok((for_input, for_grid))
}
