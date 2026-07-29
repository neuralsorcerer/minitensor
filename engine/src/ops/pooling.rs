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

use crate::{
    autograd::{AvgPool2dBackward, MaxPool2dBackward, add_to_graph},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
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
    // padding, which has no defined maximum; PyTorch rejects it the same way.
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

            values
                .par_chunks_mut(plane_out)
                .zip(indices.par_chunks_mut(plane_out))
                .enumerate()
                .for_each(|(plane, (out_values, out_indices))| {
                    let base = plane * plane_in;
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
                });

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

            values
                .par_chunks_mut(plane_out)
                .enumerate()
                .for_each(|(plane, out_values)| {
                    let base = plane * plane_in;
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
                            // cells count towards the divisor, matching PyTorch.
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
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
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
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
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
