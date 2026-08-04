// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{Conv2dBackward, add_to_graph},
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;

/// The floating-point types convolution is implemented for.
///
/// Convolution lowers to im2col + GEMM in both directions, so the only things
/// that vary with the dtype are the slice accessors and which BLAS entry point
/// to call. Capturing that in one trait lets the forward and both gradient
/// kernels be written once instead of once per dtype -- which is why they were
/// f32-only: nobody wanted to copy them.
pub(crate) trait ConvScalar: Copy + Default + Send + Sync + std::ops::AddAssign {
    const DTYPE: DataType;
    fn slice(data: &TensorData) -> Option<&[Self]>;
    fn into_tensor_data(values: Vec<Self>, device: Device) -> TensorData;
    /// Row-major `C[m, n] = A[m, k] @ B[k, n]`.
    ///
    /// # Safety
    /// `a`, `b` and `c` must point to contiguous row-major buffers of at least
    /// `m * k`, `k * n` and `m * n` elements; see [`crate::ops::linalg::gemm_f32`].
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);
    /// Row-major `C[m, n] = A[m, k]^T @ B[k, n]`, with `a` holding the logical
    /// `(m, k)` operand as `(k, m)`.
    ///
    /// The weight gradient path needs the weight as `[K, C_out]` while it is
    /// stored `[C_out, K]`. Reading it transposed by stride costs nothing;
    /// materialising it was a serial strided-write copy of the whole weight.
    ///
    /// # Safety
    /// As [`Self::gemm`], with `a` read as `k * m` elements.
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);
}

impl ConvScalar for f32 {
    const DTYPE: DataType = DataType::Float32;
    #[inline]
    fn slice(data: &TensorData) -> Option<&[Self]> {
        data.as_f32_slice()
    }
    #[inline]
    fn into_tensor_data(values: Vec<Self>, device: Device) -> TensorData {
        TensorData::from_vec_f32(values, device)
    }
    #[inline]
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self) {
        unsafe { crate::ops::linalg::gemm_f32(m, k, n, a, b, c) }
    }
    #[inline]
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self) {
        unsafe { crate::ops::linalg::gemm_tn_f32(m, k, n, a, b, c) }
    }
}

impl ConvScalar for f64 {
    const DTYPE: DataType = DataType::Float64;
    #[inline]
    fn slice(data: &TensorData) -> Option<&[Self]> {
        data.as_f64_slice()
    }
    #[inline]
    fn into_tensor_data(values: Vec<Self>, device: Device) -> TensorData {
        TensorData::from_vec_f64(values, device)
    }
    #[inline]
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self) {
        unsafe { crate::ops::linalg::gemm_f64(m, k, n, a, b, c) }
    }
    #[inline]
    unsafe fn gemm_tn(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self) {
        unsafe { crate::ops::linalg::gemm_tn_f64(m, k, n, a, b, c) }
    }
}

/// Perform 1D convolution on the input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, L]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in, K]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution
/// * `padding` - Zero padding added to both ends of the input
///
/// Implemented by giving the signal a singleton height and deferring to
/// [`conv2d`]. A dedicated kernel would avoid the two reshapes, but it would
/// also duplicate the windowing arithmetic and its backward pass — the part
/// most likely to be got wrong — for no behavioural gain. The reshapes are
/// themselves autograd-aware, so gradients flow without any new plumbing.
///
/// Inherits `conv2d`'s restriction to CPU tensors.
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    if input.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "conv1d expects 3D input tensor [N, C_in, L]",
        ));
    }
    if weight.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "conv1d expects 3D weight tensor [C_out, C_in, K]",
        ));
    }
    // Checked here rather than left to `conv2d`, whose message would name an
    // operation the caller never invoked.
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "conv1d is implemented only for floating point tensors",
        ));
    }

    let input_dims = input.shape().dims().to_vec();
    let weight_dims = weight.shape().dims().to_vec();

    let input_2d = input.reshape(Shape::new(vec![
        input_dims[0],
        input_dims[1],
        1,
        input_dims[2],
    ]))?;
    let weight_2d = weight.reshape(Shape::new(vec![
        weight_dims[0],
        weight_dims[1],
        1,
        weight_dims[2],
    ]))?;

    let output = conv2d(&input_2d, &weight_2d, bias, (1, stride), (0, padding))?;
    let out_dims = output.shape().dims().to_vec();
    output.reshape(Shape::new(vec![out_dims[0], out_dims[1], out_dims[3]]))
}

/// Perform 2D convolution on the input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, H, W]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in, kH, kW]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution `(sH, sW)`
/// * `padding` - Zero padding added to both sides of the input `(pH, pW)`
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    // Validate dimensions
    if input.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv2d expects 4D input tensor [N, C_in, H, W]",
        ));
    }
    if weight.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv2d expects 4D weight tensor [C_out, C_in, kH, kW]",
        ));
    }

    let batch_size = input.size(0)?;
    let in_channels = input.size(1)?;
    let input_height = input.size(2)?;
    let input_width = input.size(3)?;

    let out_channels = weight.size(0)?;
    let weight_in_channels = weight.size(1)?;
    let kernel_h = weight.size(2)?;
    let kernel_w = weight.size(3)?;

    if in_channels != weight_in_channels {
        return Err(MinitensorError::shape_mismatch(
            vec![weight_in_channels],
            vec![in_channels],
        ));
    }

    if let Some(b) = bias
        && (b.ndim() != 1 || b.size(0)? != out_channels)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![out_channels],
            vec![b.size(0)?],
        ));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(MinitensorError::invalid_operation(
            "stride values must be greater than zero",
        ));
    }

    if kernel_h > input_height + 2 * padding.0 || kernel_w > input_width + 2 * padding.1 {
        return Err(MinitensorError::invalid_operation(
            "kernel size cannot be larger than padded input",
        ));
    }

    let output_height = (input_height + 2 * padding.0 - kernel_h) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_w) / stride.1 + 1;
    let output_shape = Shape::new(vec![batch_size, out_channels, output_height, output_width]);

    if !input.device().is_cpu() || !weight.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "conv2d is implemented only for CPU tensors",
        ));
    }
    if weight.dtype() != input.dtype() || bias.is_some_and(|b| b.dtype() != input.dtype()) {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", input.dtype()),
            format!("{:?}", weight.dtype()),
        ));
    }

    let output_data = match input.dtype() {
        DataType::Float32 => conv2d_forward::<f32>(
            input,
            weight,
            bias,
            ConvGeometry {
                batch_size,
                in_channels,
                input_height,
                input_width,
                out_channels,
                kernel_h,
                kernel_w,
                output_height,
                output_width,
                stride,
                padding,
            },
        )?,
        DataType::Float64 => conv2d_forward::<f64>(
            input,
            weight,
            bias,
            ConvGeometry {
                batch_size,
                in_channels,
                input_height,
                input_width,
                out_channels,
                kernel_h,
                kernel_w,
                output_height,
                output_width,
                stride,
                padding,
            },
        )?,
        _ => {
            return Err(MinitensorError::invalid_operation(
                "conv2d is implemented only for floating point tensors",
            ));
        }
    };

    let requires_grad =
        input.requires_grad() || weight.requires_grad() || bias.is_some_and(|b| b.requires_grad());
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        input.dtype(),
        input.device(),
        requires_grad,
    );

    if requires_grad {
        let mut deps: SmallVec<[_; 3]> = SmallVec::new();
        if input.requires_grad() {
            deps.push(input.id());
        }
        if weight.requires_grad() {
            deps.push(weight.id());
        }
        let bias_requires_grad = bias.is_some_and(|b| b.requires_grad());
        if bias_requires_grad {
            deps.push(bias.unwrap().id());
        }
        let grad_fn = Arc::new(Conv2dBackward {
            input: input.detach(),
            weight: weight.detach(),
            input_id: input.id(),
            weight_id: weight.id(),
            bias_id: bias.map(|b| b.id()),
            input_requires_grad: input.requires_grad(),
            weight_requires_grad: weight.requires_grad(),
            bias_requires_grad,
            stride,
            padding,
            deps,
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// The shape parameters a convolution kernel needs, gathered so the generic
/// body does not take a dozen positional arguments.
pub(crate) struct ConvGeometry {
    pub batch_size: usize,
    pub in_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

/// The half-open range of output positions along one axis whose input
/// coordinate `o*stride + k_off - pad` lands inside `[0, dim)`.
///
/// Padding only ever clips a prefix and a suffix of the axis, so hoisting this
/// out of the element loop removes the per-element bounds test as well as the
/// index arithmetic behind it.
#[inline]
pub(crate) fn in_bounds_range(
    k_off: usize,
    pad: usize,
    dim: usize,
    stride: usize,
    out: usize,
) -> (usize, usize) {
    let lo = if pad > k_off {
        (pad - k_off).div_ceil(stride).min(out)
    } else {
        0
    };
    let hi = if dim + pad > k_off {
        (dim + pad - k_off).div_ceil(stride).min(out)
    } else {
        0
    };
    (lo, hi.max(lo))
}

/// im2col + GEMM forward pass, for one element type.
///
/// Lower each output position's receptive field into a column of `cols`
/// (`[K, N*out_h*out_w]`, `K = C_in*kH*kW`), then a single matrix multiply
/// `weight[C_out, K] @ cols` produces `[C_out, N*out_h*out_w]`, which is
/// scattered (with bias) into the `[N, C_out, out_h, out_w]` output. `weight`
/// is already laid out as `[C_out, K]`, so it needs no repacking. This routes
/// the arithmetic through the tuned GEMM instead of a naive per-output
/// accumulation, and produces the same cross-correlation result.
fn conv2d_forward<T: ConvScalar>(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    geom: ConvGeometry,
) -> Result<TensorData> {
    let ConvGeometry {
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
    } = geom;

    let input_data = T::slice(input.data())
        .ok_or_else(|| MinitensorError::invalid_operation("Expected float input data"))?;
    let weight_data = T::slice(weight.data())
        .ok_or_else(|| MinitensorError::invalid_operation("Expected float weight data"))?;
    let bias_data = match bias {
        Some(bias) => Some(
            T::slice(bias.data())
                .ok_or_else(|| MinitensorError::invalid_operation("Expected float bias data"))?,
        ),
        None => None,
    };

    let ohw = output_height * output_width;
    let k_dim = in_channels * kernel_h * kernel_w;
    let n_cols = batch_size * ohw;
    let kh_kw = kernel_h * kernel_w;

    let mut output_vec = vec![T::default(); batch_size * out_channels * ohw];

    if !output_vec.is_empty() {
        // Build cols row by row (one row per kernel-input index `k`), so each
        // row is written contiguously.
        //
        // The output position is walked with nested loops rather than recovered
        // from a flat counter. Decomposing the counter needed four integer
        // divisions per element -- by `ohw` and `output_width`, both runtime
        // values, so they stay real divisions -- across 4.7M elements for a
        // 16x32x32x32 conv. Walking `n`, `oh`, `ow` costs none.
        //
        // The in-bounds range of output positions is also computed once per
        // row instead of testing each element: padding only ever clips a prefix
        // and a suffix, and `cols` starts zeroed, so the pad needs no writing at
        // all. What is left is a contiguous copy per row when the horizontal
        // stride is 1, which is the overwhelmingly common case.
        let mut cols = vec![T::default(); k_dim * n_cols];
        cols.par_chunks_mut(n_cols)
            .enumerate()
            .for_each(|(k, row)| {
                let ic = k / kh_kw;
                let rem = k % kh_kw;
                let ky = rem / kernel_w;
                let kx = rem % kernel_w;
                let (oh_lo, oh_hi) =
                    in_bounds_range(ky, padding.0, input_height, stride.0, output_height);
                let (ow_lo, ow_hi) =
                    in_bounds_range(kx, padding.1, input_width, stride.1, output_width);
                if oh_lo >= oh_hi || ow_lo >= ow_hi {
                    return;
                }
                let span = ow_hi - ow_lo;
                for n in 0..batch_size {
                    let dst_n = n * ohw;
                    let plane = (n * in_channels + ic) * input_height;
                    for oh in oh_lo..oh_hi {
                        let ih = oh * stride.0 + ky - padding.0;
                        let src = (plane + ih) * input_width;
                        let dst = dst_n + oh * output_width + ow_lo;
                        if stride.1 == 1 {
                            let s = src + ow_lo + kx - padding.1;
                            row[dst..dst + span].copy_from_slice(&input_data[s..s + span]);
                        } else {
                            for (i, slot) in row[dst..dst + span].iter_mut().enumerate() {
                                let iw = (ow_lo + i) * stride.1 + kx - padding.1;
                                *slot = input_data[src + iw];
                            }
                        }
                    }
                }
            });

        let mut gemm_out = vec![T::default(); out_channels * n_cols];
        // SAFETY: `weight_data` is [C_out, k_dim], `cols` is [k_dim, n_cols],
        // and `gemm_out` is [C_out, n_cols]; all contiguous row-major with
        // matching dimensions.
        unsafe {
            T::gemm(
                out_channels,
                k_dim,
                n_cols,
                weight_data.as_ptr(),
                cols.as_ptr(),
                gemm_out.as_mut_ptr(),
            );
        }

        // Scatter [C_out, N*ohw] into [N, C_out, ohw], adding bias. For a given
        // (n, oc) the source and destination are contiguous `ohw` slabs.
        output_vec
            .par_chunks_mut(ohw)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let n = chunk_idx / out_channels;
                let oc = chunk_idx % out_channels;
                let base = oc * n_cols + n * ohw;
                for (o, &v) in out_chunk.iter_mut().zip(&gemm_out[base..base + ohw]) {
                    *o = v;
                    if let Some(bd) = bias_data {
                        *o += bd[oc];
                    }
                }
            });
    }

    Ok(T::into_tensor_data(output_vec, input.device()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor, TensorData},
    };

    #[test]
    fn test_conv2d_basic() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1., 2., 3., 4.],
                Device::cpu(),
            )),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.], Device::cpu())),
            Shape::new(vec![1, 1, 1, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let bias = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0)).unwrap();
        let data = out.data().as_f32_slice().unwrap();
        assert_eq!(data, &[2., 3., 4., 5.]);
    }

    #[test]
    fn test_conv2d_padding_and_stride() {
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(input_data, Device::cpu())),
            Shape::new(vec![1, 1, 4, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1., 0., 0., 1.],
                Device::cpu(),
            )),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = conv2d(&input, &weight, None, (2, 2), (1, 1)).unwrap();
        assert_eq!(out.shape(), &Shape::new(vec![1, 1, 3, 3]));
        let data = out.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1., 3., 0., 9., 17., 8., 0., 14., 16.]);
    }

    #[test]
    fn test_conv2d_invalid_kernel() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![0.; 4], Device::cpu())),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![0.; 25], Device::cpu())),
            Shape::new(vec![1, 1, 5, 5]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert!(result.is_err());
    }
}
