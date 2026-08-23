// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Two-dimensional pooling.
//!
//! Both kernels parallelise over the flattened `[N, C]` plane so each task owns
//! a disjoint output slab, and both walk the same padded coordinate mapping the
//! convolution uses: an output position `(oh, ow)` reads the window starting at
//! `(oh * stride - padding, ow * stride - padding)`, with out-of-range
//! coordinates treated as padding rather than clamped.

use crate::autograd::with_grad_fn;
use crate::ops::map::{par_out_chunks, par_out_chunks2};
use crate::{
    autograd::{AvgPool2dBackward, MaxPool2dBackward},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// Output extent of a pooling axis, matching the convolution's formula.
fn pooled_extent(input: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    let padded = input + 2 * padding;
    if kernel == 0 {
        return Err(MinitensorError::invalid_argument(
            "pooling kernel size must be non-zero",
        ));
    }
    if stride == 0 {
        return Err(MinitensorError::invalid_argument(
            "pooling stride must be non-zero",
        ));
    }
    if padded < kernel {
        return Err(MinitensorError::invalid_argument(format!(
            "pooling window {kernel} is larger than the padded input extent {padded}"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

/// Shared validation and geometry for the 2-D poolers.
struct PoolGeometry {
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
}

fn pool_geometry(
    input: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    name: &str,
) -> Result<PoolGeometry> {
    if input.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} expects a 4D input tensor [N, C, H, W]"
        )));
    }
    // A window that starts entirely inside the padding would pool nothing but
    // padding, which has no defined maximum, so it is rejected.
    if padding.0 * 2 > kernel.0 || padding.1 * 2 > kernel.1 {
        return Err(MinitensorError::invalid_argument(format!(
            "{name} padding must not exceed half the window size"
        )));
    }
    let (batch, channels) = (input.size(0)?, input.size(1)?);
    let (in_h, in_w) = (input.size(2)?, input.size(3)?);
    Ok(PoolGeometry {
        batch,
        channels,
        in_h,
        in_w,
        out_h: pooled_extent(in_h, kernel.0, stride.0, padding.0)?,
        out_w: pooled_extent(in_w, kernel.1, stride.1, padding.1)?,
    })
}

/// Generates the forward kernel for one float dtype.
///
/// `indices` records, for every output element, the flat offset within its
/// `[H, W]` plane that supplied the maximum, so the backward pass can scatter
/// straight back without re-reading the input.
macro_rules! max_pool2d_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        fn $name(
            input: &Tensor,
            geometry: &PoolGeometry,
            kernel: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
        ) -> Result<(Vec<$ty>, Vec<i64>)> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("max_pool2d received a mismatched dtype")
            })?;
            let PoolGeometry {
                batch,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
            } = *geometry;

            let plane_out = out_h * out_w;
            let plane_in = in_h * in_w;
            let mut values = vec![<$ty>::NAN; batch * channels * plane_out];
            let mut indices = vec![0i64; batch * channels * plane_out];

            par_out_chunks2(
                &mut values,
                &mut indices,
                plane_out,
                &|first, out_values, out_indices| {
                    let base = (first / plane_out) * plane_in;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut best = <$ty>::NEG_INFINITY;
                            let mut best_index = -1i64;
                            let mut saw_nan = false;
                            for ky in 0..kernel.0 {
                                let ih = oh * stride.0 + ky;
                                if ih < padding.0 || ih >= in_h + padding.0 {
                                    continue;
                                }
                                let ih = ih - padding.0;
                                for kx in 0..kernel.1 {
                                    let iw = ow * stride.1 + kx;
                                    if iw < padding.1 || iw >= in_w + padding.1 {
                                        continue;
                                    }
                                    let iw = iw - padding.1;
                                    let offset = ih * in_w + iw;
                                    let value = data[base + offset];
                                    // NaN wins, as it does in `max`: the first
                                    // one encountered takes the window.
                                    if value != value {
                                        if !saw_nan {
                                            saw_nan = true;
                                            best = value;
                                            best_index = offset as i64;
                                        }
                                    } else if !saw_nan && (best_index < 0 || value > best) {
                                        best = value;
                                        best_index = offset as i64;
                                    }
                                }
                            }
                            let slot = oh * out_w + ow;
                            out_values[slot] = best;
                            out_indices[slot] = best_index;
                        }
                    }
                },
            );

            Ok((values, indices))
        }
    };
}

max_pool2d_kernel!(max_pool2d_f32, f32, as_f32_slice);
max_pool2d_kernel!(max_pool2d_f64, f64, as_f64_slice);

macro_rules! avg_pool2d_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        fn $name(
            input: &Tensor,
            geometry: &PoolGeometry,
            kernel: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
            count_include_pad: bool,
        ) -> Result<Vec<$ty>> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("avg_pool2d received a mismatched dtype")
            })?;
            let PoolGeometry {
                batch,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
            } = *geometry;

            let plane_out = out_h * out_w;
            let plane_in = in_h * in_w;
            let mut values = vec![0 as $ty; batch * channels * plane_out];

            par_out_chunks(&mut values, plane_out, &|first, out_values| {
                let base = (first / plane_out) * plane_in;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut total = 0 as $ty;
                        let mut counted = 0usize;
                        for ky in 0..kernel.0 {
                            let ih = oh * stride.0 + ky;
                            let inside_h = ih >= padding.0 && ih < in_h + padding.0;
                            for kx in 0..kernel.1 {
                                let iw = ow * stride.1 + kx;
                                let inside_w = iw >= padding.1 && iw < in_w + padding.1;
                                if inside_h && inside_w {
                                    let offset = (ih - padding.0) * in_w + (iw - padding.1);
                                    total += data[base + offset];
                                    counted += 1;
                                }
                            }
                        }
                        // `count_include_pad` decides whether the padded
                        // cells count towards the divisor.
                        let divisor = if count_include_pad {
                            kernel.0 * kernel.1
                        } else {
                            counted
                        };
                        let slot = oh * out_w + ow;
                        out_values[slot] = if divisor == 0 {
                            0 as $ty
                        } else {
                            total / divisor as $ty
                        };
                    }
                }
            });

            Ok(values)
        }
    };
}

avg_pool2d_kernel!(avg_pool2d_f32, f32, as_f32_slice);
avg_pool2d_kernel!(avg_pool2d_f64, f64, as_f64_slice);

/// The half-open input range that output position `index` pools over.
///
/// This is the whole of what makes a pooling "adaptive": the window comes from
/// the ratio of the two extents rather than from a fixed kernel, so it stretches
/// to cover whatever input it is given, and neighbouring windows overlap or
/// leave gaps by however much they must. `start` rounds down and `end` rounds
/// up, which is what guarantees every input position falls in at least one
/// window and no window is empty.
///
/// When `out_size` divides `in_size` this degenerates exactly to a regular pool
/// with kernel and stride both `in_size / out_size` -- which is the case a
/// caller can check against, and the reason the formula has to be this one and
/// not a plausible neighbour.
#[inline]
pub(crate) fn adaptive_window(
    index: usize,
    in_size: usize,
    out_size: usize,
) -> std::ops::Range<usize> {
    let start = index * in_size / out_size;
    let end = ((index + 1) * in_size).div_ceil(out_size);
    start..end
}

/// Validate an adaptive pooling request and produce its geometry.
fn adaptive_geometry(
    input: &Tensor,
    output_size: (usize, usize),
    name: &str,
) -> Result<PoolGeometry> {
    if input.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} expects a 4D input tensor [N, C, H, W]"
        )));
    }
    let (batch, channels) = (input.size(0)?, input.size(1)?);
    let (in_h, in_w) = (input.size(2)?, input.size(3)?);
    // An empty axis has no values to pool and the window over it would be
    // empty, so an average would be zero divided by zero. Asking for no output
    // along that axis is fine; asking for some is not.
    if (in_h == 0 && output_size.0 > 0) || (in_w == 0 && output_size.1 > 0) {
        return Err(MinitensorError::invalid_argument(format!(
            "{name} cannot pool an empty spatial axis into a non-empty one"
        )));
    }
    Ok(PoolGeometry {
        batch,
        channels,
        in_h,
        in_w,
        out_h: output_size.0,
        out_w: output_size.1,
    })
}

macro_rules! adaptive_avg_pool2d_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        fn $name(input: &Tensor, geometry: &PoolGeometry) -> Result<Vec<$ty>> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("adaptive_avg_pool2d received a mismatched dtype")
            })?;
            let PoolGeometry {
                batch,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
            } = *geometry;

            let plane_out = out_h * out_w;
            let plane_in = in_h * in_w;
            let mut values = vec![0 as $ty; batch * channels * plane_out];
            if plane_out == 0 {
                return Ok(values);
            }

            par_out_chunks(&mut values, plane_out, &|first, out_values| {
                let base = (first / plane_out) * plane_in;
                for oh in 0..out_h {
                    let rows = adaptive_window(oh, in_h, out_h);
                    for ow in 0..out_w {
                        let cols = adaptive_window(ow, in_w, out_w);
                        let count = rows.len() * cols.len();
                        let mut total = 0 as $ty;
                        for ih in rows.clone() {
                            let row = base + ih * in_w;
                            for value in &data[row + cols.start..row + cols.end] {
                                total += *value;
                            }
                        }
                        out_values[oh * out_w + ow] = total / count as $ty;
                    }
                }
            });
            Ok(values)
        }
    };
}

adaptive_avg_pool2d_kernel!(adaptive_avg_pool2d_f32, f32, as_f32_slice);
adaptive_avg_pool2d_kernel!(adaptive_avg_pool2d_f64, f64, as_f64_slice);

macro_rules! adaptive_max_pool2d_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        fn $name(input: &Tensor, geometry: &PoolGeometry) -> Result<(Vec<$ty>, Vec<i64>)> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("adaptive_max_pool2d received a mismatched dtype")
            })?;
            let PoolGeometry {
                batch,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
            } = *geometry;

            let plane_out = out_h * out_w;
            let plane_in = in_h * in_w;
            let mut values = vec![0 as $ty; batch * channels * plane_out];
            let mut indices = vec![0i64; batch * channels * plane_out];
            if plane_out == 0 {
                return Ok((values, indices));
            }

            par_out_chunks2(
                &mut values,
                &mut indices,
                plane_out,
                &|first, out_values, out_indices| {
                    let base = (first / plane_out) * plane_in;
                    for oh in 0..out_h {
                        let rows = adaptive_window(oh, in_h, out_h);
                        for ow in 0..out_w {
                            let cols = adaptive_window(ow, in_w, out_w);
                            let mut best = <$ty>::NEG_INFINITY;
                            let mut best_index = -1i64;
                            let mut saw_nan = false;
                            for ih in rows.clone() {
                                for iw in cols.clone() {
                                    let offset = ih * in_w + iw;
                                    let value = data[base + offset];
                                    // A NaN in the window wins, as it does in
                                    // `max_pool2d` and in `max` itself: the
                                    // maximum of a set containing one is not a
                                    // number, and quietly skipping it would let
                                    // a NaN vanish from a network.
                                    if value.is_nan() {
                                        if !saw_nan {
                                            saw_nan = true;
                                            best = value;
                                            best_index = offset as i64;
                                        }
                                    } else if !saw_nan && value > best {
                                        best = value;
                                        best_index = offset as i64;
                                    }
                                }
                            }
                            let slot = oh * out_w + ow;
                            out_values[slot] = best;
                            out_indices[slot] = best_index;
                        }
                    }
                },
            );
            Ok((values, indices))
        }
    };
}

adaptive_max_pool2d_kernel!(adaptive_max_pool2d_f32, f32, as_f32_slice);
adaptive_max_pool2d_kernel!(adaptive_max_pool2d_f64, f64, as_f64_slice);

/// Average pooling to a fixed output size, whatever the input size is.
///
/// A regular pool needs a kernel and a stride, so the caller has to know the
/// input's spatial extent to hit a chosen output extent -- and cannot hit one at
/// all when the extent does not divide evenly, because a fixed window cannot
/// produce unequal groups. This derives the window from the ratio instead, which
/// is what lets a classifier head take any input size and still hand the linear
/// layer the shape it expects.
///
/// `output_size` of `(1, 1)` is the global average pool that ends most
/// convolutional networks.
pub fn adaptive_avg_pool2d(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
    let geometry = adaptive_geometry(input, output_size, "adaptive_avg_pool2d")?;
    let out_shape = Shape::new(vec![
        geometry.batch,
        geometry.channels,
        geometry.out_h,
        geometry.out_w,
    ]);

    let data = match input.dtype() {
        DataType::Float32 => {
            TensorData::from_vec_f32(adaptive_avg_pool2d_f32(input, &geometry)?, input.device())
        }
        DataType::Float64 => {
            TensorData::from_vec_f64(adaptive_avg_pool2d_f64(input, &geometry)?, input.device())
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "adaptive_avg_pool2d only supports float32 and float64 tensors",
            ));
        }
    };

    let mut output = Tensor::new(
        Arc::new(data),
        out_shape,
        input.dtype(),
        input.device(),
        input.requires_grad(),
    );

    if input.requires_grad() {
        let grad_fn = Arc::new(crate::autograd::AdaptiveAvgPool2dBackward {
            input_id: input.id(),
            input_shape: input.shape().dims().to_vec(),
            output_size,
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}

/// Max pooling to a fixed output size. See [`adaptive_avg_pool2d`] for what
/// "adaptive" buys.
///
/// The winning positions are retained internally, so the backward pass is the
/// same scatter a regular max pool performs and shares its implementation --
/// which is the whole reason this records them rather than recomputing the
/// windows.
pub fn adaptive_max_pool2d(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
    let geometry = adaptive_geometry(input, output_size, "adaptive_max_pool2d")?;
    let out_shape = Shape::new(vec![
        geometry.batch,
        geometry.channels,
        geometry.out_h,
        geometry.out_w,
    ]);

    let (data, indices) = match input.dtype() {
        DataType::Float32 => {
            let (values, indices) = adaptive_max_pool2d_f32(input, &geometry)?;
            (TensorData::from_vec_f32(values, input.device()), indices)
        }
        DataType::Float64 => {
            let (values, indices) = adaptive_max_pool2d_f64(input, &geometry)?;
            (TensorData::from_vec_f64(values, input.device()), indices)
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "adaptive_max_pool2d only supports float32 and float64 tensors",
            ));
        }
    };

    let mut output = Tensor::new(
        Arc::new(data),
        out_shape,
        input.dtype(),
        input.device(),
        input.requires_grad(),
    );

    if input.requires_grad() {
        let grad_fn = Arc::new(MaxPool2dBackward {
            input_id: input.id(),
            input_shape: input.shape().dims().to_vec(),
            indices,
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}

/// 1-D adaptive average pooling over `[N, C, L]`.
pub fn adaptive_avg_pool1d(input: &Tensor, output_size: usize) -> Result<Tensor> {
    adaptive_pool1d(input, output_size, "adaptive_avg_pool1d", |t, size| {
        adaptive_avg_pool2d(t, size)
    })
}

/// 1-D adaptive max pooling over `[N, C, L]`.
pub fn adaptive_max_pool1d(input: &Tensor, output_size: usize) -> Result<Tensor> {
    adaptive_pool1d(input, output_size, "adaptive_max_pool1d", |t, size| {
        adaptive_max_pool2d(t, size)
    })
}

/// Give a 1-D signal a singleton height and defer to the 2-D pooler, as the
/// fixed-window 1-D poolers do -- one window rule and one backward pass rather
/// than two to keep in step.
fn adaptive_pool1d<F>(input: &Tensor, output_size: usize, name: &str, pool: F) -> Result<Tensor>
where
    F: Fn(&Tensor, (usize, usize)) -> Result<Tensor>,
{
    if input.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} expects a 3D input tensor [N, C, L]"
        )));
    }
    let dims = input.shape().dims().to_vec();
    let widened = input.reshape(Shape::new(vec![dims[0], dims[1], 1, dims[2]]))?;
    let pooled = pool(&widened, (1, output_size))?;
    let out = pooled.shape().dims().to_vec();
    pooled.reshape(Shape::new(vec![out[0], out[1], out[3]]))
}

/// Max pooling over a 4-D `[N, C, H, W]` input.
///
/// Returns the pooled values; the winning positions are retained internally for
/// the backward pass.
pub fn max_pool2d(
    input: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    let geometry = pool_geometry(input, kernel, stride, padding, "max_pool2d")?;
    let out_shape = Shape::new(vec![
        geometry.batch,
        geometry.channels,
        geometry.out_h,
        geometry.out_w,
    ]);

    let (data, indices) = match input.dtype() {
        DataType::Float32 => {
            let (values, indices) = max_pool2d_f32(input, &geometry, kernel, stride, padding)?;
            (TensorData::from_vec_f32(values, input.device()), indices)
        }
        DataType::Float64 => {
            let (values, indices) = max_pool2d_f64(input, &geometry, kernel, stride, padding)?;
            (TensorData::from_vec_f64(values, input.device()), indices)
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "max_pool2d only supports float32 and float64 tensors",
            ));
        }
    };

    let mut output = Tensor::new(
        Arc::new(data),
        out_shape,
        input.dtype(),
        input.device(),
        input.requires_grad(),
    );

    if input.requires_grad() {
        let grad_fn = Arc::new(MaxPool2dBackward {
            input_id: input.id(),
            input_shape: input.shape().dims().to_vec(),
            indices,
        });
        output = with_grad_fn(output, grad_fn)?;
    }

    Ok(output)
}

/// Average pooling over a 4-D `[N, C, H, W]` input.
///
/// `count_include_pad` selects whether padded cells contribute to the divisor.
pub fn avg_pool2d(
    input: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
) -> Result<Tensor> {
    let geometry = pool_geometry(input, kernel, stride, padding, "avg_pool2d")?;
    let out_shape = Shape::new(vec![
        geometry.batch,
        geometry.channels,
        geometry.out_h,
        geometry.out_w,
    ]);

    let data = match input.dtype() {
        DataType::Float32 => TensorData::from_vec_f32(
            avg_pool2d_f32(input, &geometry, kernel, stride, padding, count_include_pad)?,
            input.device(),
        ),
        DataType::Float64 => TensorData::from_vec_f64(
            avg_pool2d_f64(input, &geometry, kernel, stride, padding, count_include_pad)?,
            input.device(),
        ),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "avg_pool2d only supports float32 and float64 tensors",
            ));
        }
    };

    let mut output = Tensor::new(
        Arc::new(data),
        out_shape,
        input.dtype(),
        input.device(),
        input.requires_grad(),
    );

    if input.requires_grad() {
        let grad_fn = Arc::new(AvgPool2dBackward {
            input_id: input.id(),
            input_shape: input.shape().dims().to_vec(),
            kernel,
            stride,
            padding,
            count_include_pad,
        });
        output = with_grad_fn(output, grad_fn)?;
    }

    Ok(output)
}

/// Reshape a `[N, C, L]` signal to `[N, C, 1, L]`, apply a 2-D pooling kernel
/// with a singleton height, and drop the height again.
///
/// Sharing the 2-D kernels keeps one implementation of the window arithmetic
/// and one backward pass rather than two that must be kept in agreement.
fn pool1d(
    input: &Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
    name: &str,
    apply: impl Fn(&Tensor, (usize, usize), (usize, usize), (usize, usize)) -> Result<Tensor>,
) -> Result<Tensor> {
    if input.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} expects a 3D input tensor [N, C, L]"
        )));
    }
    let dims = input.shape().dims().to_vec();
    let widened = input.reshape(Shape::new(vec![dims[0], dims[1], 1, dims[2]]))?;
    let pooled = apply(&widened, (1, kernel), (1, stride), (0, padding))?;
    let out = pooled.shape().dims().to_vec();
    pooled.reshape(Shape::new(vec![out[0], out[1], out[3]]))
}

/// 1-D max pooling over `[N, C, L]`.
///
/// As with [`max_pool2d`], `stride` is the caller's to choose but conventionally
/// equals `kernel`; the wrappers in `nn` apply that default.
pub fn max_pool1d(input: &Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
    pool1d(input, kernel, stride, padding, "max_pool1d", max_pool2d)
}

/// 1-D average pooling over `[N, C, L]`.
pub fn avg_pool1d(
    input: &Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> Result<Tensor> {
    pool1d(
        input,
        kernel,
        stride,
        padding,
        "avg_pool1d",
        |t, k, s, p| avg_pool2d(t, k, s, p, count_include_pad),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use std::sync::Arc;

    fn signal(values: Vec<f64>) -> Tensor {
        let len = values.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(values, Device::cpu())),
            Shape::new(vec![1, 1, len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_pool1d_matches_a_direct_window_reference() {
        let input = signal(vec![1.0, 5.0, 2.0, 8.0, 3.0, 4.0]);

        let maxed = max_pool1d(&input, 2, 2, 0).unwrap();
        assert_eq!(maxed.shape().dims(), &[1, 1, 3]);
        assert_eq!(maxed.data().as_f64_slice().unwrap(), &[5.0, 8.0, 4.0]);

        let averaged = avg_pool1d(&input, 2, 2, 0, true).unwrap();
        assert_eq!(averaged.data().as_f64_slice().unwrap(), &[3.0, 5.0, 3.5]);

        // Stride below the kernel makes the windows overlap.
        let overlapped = max_pool1d(&input, 3, 1, 0).unwrap();
        assert_eq!(overlapped.shape().dims(), &[1, 1, 4]);
        assert_eq!(
            overlapped.data().as_f64_slice().unwrap(),
            &[5.0, 8.0, 8.0, 8.0]
        );
    }

    #[test]
    fn test_avg_pool1d_count_include_pad_picks_the_divisor() {
        let input = signal(vec![1.0, 2.0, 3.0, 4.0]);

        // The last window covers cells 3 and 4 plus one padding cell.
        let included = avg_pool1d(&input, 3, 1, 1, true).unwrap();
        let last = *included.data().as_f64_slice().unwrap().last().unwrap();
        assert!((last - 7.0 / 3.0).abs() < 1e-12, "{last}");

        let excluded = avg_pool1d(&input, 3, 1, 1, false).unwrap();
        let last = *excluded.data().as_f64_slice().unwrap().last().unwrap();
        assert!((last - 3.5).abs() < 1e-12, "{last}");
    }

    #[test]
    fn test_pool1d_rejects_wrong_rank() {
        let flat = Tensor::new(
            Arc::new(TensorData::from_vec_f64(vec![0.0; 4], Device::cpu())),
            Shape::new(vec![2, 2]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        assert!(max_pool1d(&flat, 2, 2, 0).is_err());
        assert!(avg_pool1d(&flat, 2, 2, 0, true).is_err());
    }
}
