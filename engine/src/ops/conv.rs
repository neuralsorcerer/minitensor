// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::with_grad_fn;
use crate::ops::map::par_out_chunks;
use crate::{
    autograd::Conv2dBackward,
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
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
    dilation: usize,
    groups: usize,
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

    let output = conv2d(
        &input_2d,
        &weight_2d,
        bias,
        (1, stride),
        (0, padding),
        (1, dilation),
        groups,
    )?;
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
    dilation: (usize, usize),
    groups: usize,
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

    if groups == 0 {
        return Err(MinitensorError::invalid_operation(
            "groups must be greater than zero",
        ));
    }
    if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(MinitensorError::invalid_operation(format!(
            "conv2d groups={groups} must divide both in_channels={in_channels} and out_channels={out_channels}"
        )));
    }
    // Each group's kernel spans only its own slice of the input channels, so
    // the weight is `[C_out, C_in / groups, kH, kW]`. At `groups == 1` this is
    // the ordinary check.
    if in_channels / groups != weight_in_channels {
        return Err(MinitensorError::shape_mismatch(
            vec![weight_in_channels],
            vec![in_channels / groups],
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
    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(MinitensorError::invalid_operation(
            "dilation values must be greater than zero",
        ));
    }

    // What the kernel actually covers once its taps are spread apart. Every
    // bound below is in terms of this rather than the kernel size, which is
    // what makes dilation a property of the geometry and not of the loops.
    let span_h = dilation.0 * (kernel_h - 1) + 1;
    let span_w = dilation.1 * (kernel_w - 1) + 1;

    if span_h > input_height + 2 * padding.0 || span_w > input_width + 2 * padding.1 {
        return Err(MinitensorError::invalid_operation(
            "kernel size cannot be larger than padded input",
        ));
    }

    let output_height = (input_height + 2 * padding.0 - span_h) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - span_w) / stride.1 + 1;
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
                dilation,
                groups,
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
                dilation,
                groups,
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
            dilation,
            groups,
            deps,
        });
        output = with_grad_fn(output, grad_fn)?;
    }

    Ok(output)
}

/// Transposed convolution -- the map a convolution's input gradient performs,
/// run forwards.
///
/// A convolution gathers a neighbourhood into each output position; this
/// scatters each input position back across a neighbourhood, so it grows the
/// grid where a convolution shrinks it. That is what a decoder, a generator, an
/// autoencoder or a segmentation head is built out of, and none of it could be
/// written here: padding the input and convolving is a different operation --
/// it cannot produce a stride-2 upsample at all, because the spacing between
/// contributions comes from the stride of the scatter and no amount of padding
/// puts gaps *between* input samples.
///
/// It is emphatically not a deconvolution: it does not invert a convolution's
/// values, only the shape mapping. The name is historical and the shape is the
/// part that is true.
///
/// * `input` -- `[N, C_in, H, W]`
/// * `weight` -- `[C_in, C_out / groups, kH, kW]`, input channels first, which
///   is the convolution's weight layout seen from the other side
/// * `bias` -- optional `[C_out]`
/// * `output_padding` -- extra rows and columns on the bottom and right
///
/// `H_out = (H - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1`.
///
/// `output_padding` exists because the shape mapping is not injective: with
/// `stride = s`, `s` different input sizes convolve to the same output size,
/// and this says which of them to come back to. Values from `0` to `stride - 1`
/// name exactly those, and anything larger names no convolution at all, so it
/// is refused rather than silently producing a grid with an unreachable margin.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose2d expects 4D input tensor [N, C_in, H, W]",
        ));
    }
    if weight.ndim() != 4 {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose2d expects 4D weight tensor [C_in, C_out / groups, kH, kW]",
        ));
    }
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose2d is implemented only for floating point tensors",
        ));
    }
    if !input.device().is_cpu() || !weight.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose2d is implemented only for CPU tensors",
        ));
    }
    if weight.dtype() != input.dtype() || bias.is_some_and(|b| b.dtype() != input.dtype()) {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", input.dtype()),
            format!("{:?}", weight.dtype()),
        ));
    }

    let batch_size = input.size(0)?;
    let in_channels = input.size(1)?;
    let input_height = input.size(2)?;
    let input_width = input.size(3)?;
    let kernel_h = weight.size(2)?;
    let kernel_w = weight.size(3)?;

    if groups == 0 {
        return Err(MinitensorError::invalid_operation(
            "groups must be greater than zero",
        ));
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(MinitensorError::invalid_operation(
            "stride values must be greater than zero",
        ));
    }
    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(MinitensorError::invalid_operation(
            "dilation values must be greater than zero",
        ));
    }
    // The weight's first axis is the *input* channels here, the reverse of a
    // convolution's -- it is the same tensor a convolution would use, read from
    // the side that scatters rather than the side that gathers.
    if weight.size(0)? != in_channels {
        return Err(MinitensorError::shape_mismatch(
            vec![in_channels],
            vec![weight.size(0)?],
        ));
    }
    if !in_channels.is_multiple_of(groups) {
        return Err(MinitensorError::invalid_operation(format!(
            "conv_transpose2d groups={groups} must divide in_channels={in_channels}"
        )));
    }
    let out_channels = weight.size(1)? * groups;
    if let Some(b) = bias
        && (b.ndim() != 1 || b.size(0)? != out_channels)
    {
        return Err(MinitensorError::shape_mismatch(
            vec![out_channels],
            vec![b.size(0)?],
        ));
    }
    if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
        return Err(MinitensorError::invalid_operation(format!(
            "conv_transpose2d output_padding {output_padding:?} must be smaller than stride \
             {stride:?}; a larger value names no convolution to invert"
        )));
    }

    let span_h = dilation.0 * (kernel_h - 1) + 1;
    let span_w = dilation.1 * (kernel_w - 1) + 1;
    let padded_h = (input_height.max(1) - 1) * stride.0 + span_h + output_padding.0;
    let padded_w = (input_width.max(1) - 1) * stride.1 + span_w + output_padding.1;
    if 2 * padding.0 >= padded_h || 2 * padding.1 >= padded_w {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose2d padding removes the whole output",
        ));
    }
    let output_height = padded_h - 2 * padding.0;
    let output_width = padded_w - 2 * padding.1;

    let geometry = ConvGeometry {
        batch_size,
        // The grid being written is the output, and the signal being read is
        // the input -- the reverse of a convolution, which is the whole point,
        // and why the geometry's own names read backwards here.
        in_channels: out_channels,
        input_height: output_height,
        input_width: output_width,
        out_channels: in_channels,
        kernel_h,
        kernel_w,
        output_height: input_height,
        output_width: input_width,
        stride,
        padding,
        dilation,
        groups,
    };

    let contiguous = input.contiguous()?;
    let weight_view = weight.contiguous()?;
    let output_shape = Shape::new(vec![batch_size, out_channels, output_height, output_width]);

    macro_rules! scatter {
        ($ty:ty) => {{
            let signal = <$ty as ConvScalar>::slice(contiguous.data()).ok_or_else(|| {
                MinitensorError::internal_error("conv_transpose2d: input dtype does not match")
            })?;
            let kernel = <$ty as ConvScalar>::slice(weight_view.data()).ok_or_else(|| {
                MinitensorError::internal_error("conv_transpose2d: weight dtype does not match")
            })?;
            let mut values = scatter_columns::<$ty>(signal, kernel, &geometry);
            if let Some(b) = bias {
                let bias_view = b.contiguous()?;
                let added = <$ty as ConvScalar>::slice(bias_view.data()).ok_or_else(|| {
                    MinitensorError::internal_error("conv_transpose2d: bias dtype does not match")
                })?;
                let plane = output_height * output_width;
                par_out_chunks(&mut values, plane.max(1), &|start, row| {
                    let channel = (start / plane.max(1)) % out_channels;
                    let shift = added[channel];
                    for slot in row.iter_mut() {
                        *slot += shift;
                    }
                });
            }
            <$ty as ConvScalar>::into_tensor_data(values, input.device())
        }};
    }

    let output_data = match input.dtype() {
        DataType::Float32 => scatter!(f32),
        _ => scatter!(f64),
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
        let grad_fn = Arc::new(crate::autograd::ConvTranspose2dBackward {
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
            dilation,
            groups,
            deps,
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}

/// [`conv_transpose2d`] for a 1D signal, by giving it a singleton height.
///
/// The same reasoning as [`conv1d`]: a dedicated kernel would duplicate the
/// windowing arithmetic and its backward pass -- the part most likely to be got
/// wrong -- for no behavioural gain, and the reshapes are autograd-aware so the
/// gradient needs no new plumbing.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    if input.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose1d expects 3D input tensor [N, C_in, L]",
        ));
    }
    if weight.ndim() != 3 {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose1d expects 3D weight tensor [C_in, C_out / groups, K]",
        ));
    }
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose1d is implemented only for floating point tensors",
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

    let output = conv_transpose2d(
        &input_2d,
        &weight_2d,
        bias,
        (1, stride),
        (0, padding),
        (0, output_padding),
        (1, dilation),
        groups,
    )?;
    let out_dims = output.shape().dims().to_vec();
    output.reshape(Shape::new(vec![out_dims[0], out_dims[1], out_dims[3]]))
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
    /// Spacing between the kernel taps. `1` is an ordinary convolution; larger
    /// widens the receptive field without adding parameters, which is what
    /// dilated and atrous convolutions are.
    pub dilation: (usize, usize),
    /// How many independent convolutions the channels are split into. `1` is
    /// ordinary; `in_channels` makes it depthwise, which is the half of a
    /// depthwise-separable convolution that no amount of reshaping can express
    /// without it.
    pub groups: usize,
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

/// Copy images `[first, last)` of a `[N, C, P]` signal into the `[C, (last -
/// first) * P]` layout the gradient GEMMs contract against.
///
/// One image on its own is already `[C, P]` and needs no copy, but a GEMM over
/// several of them wants the channel outermost across all of them, and that is
/// not a slice of anything. Copying a *block* is the compromise: big enough
/// that the multiply is not 64 tiny ones, bounded so the copy is never the
/// whole batch at once.
fn block_channel_major<T: ConvScalar>(
    source: &[T],
    out: &mut [T],
    first: usize,
    last: usize,
    channels: usize,
    plane: usize,
) {
    let span = (last - first) * plane;
    if span == 0 {
        return;
    }
    par_out_chunks(out, span, &|start, row| {
        let channel = start / span;
        for n in first..last {
            let src = (n * channels + channel) * plane;
            let dst = (n - first) * plane;
            row[dst..dst + plane].copy_from_slice(&source[src..src + plane]);
        }
    });
}

/// How many images to take at once, so neither the lowering nor the copied
/// signal outgrows [`COL_BLOCK_BYTES`].
fn images_per_block<T>(batch: usize, lowered: usize, signal: usize) -> usize {
    let widest = lowered.max(signal).max(1) * std::mem::size_of::<T>();
    (COL_BLOCK_BYTES / widest).clamp(1, batch.max(1))
}

/// `col2im(weight^T @ source)` -- scatter a signal back through a kernel onto
/// the larger grid it was gathered from.
///
/// This is one operation with two names. Read backwards it is a convolution's
/// gradient with respect to its input; read forwards it is a transposed
/// convolution, which is why `conv_transpose2d` is not a separate algorithm and
/// does not get a second copy of this. The two differ only in which side the
/// caller calls the input.
///
/// In `geometry`, the `input_*` fields describe the grid being written and the
/// `output_*` fields the signal being read -- the convolution's own naming,
/// kept so that a caller coming from either direction reads the same struct.
/// `weight` is `[out_channels, in_channels / groups, kH, kW]`.
///
/// The signal arrives in its natural `[N, C, H, W]` layout and is taken one
/// image at a time; within an image the scatter-add runs one channel plane per
/// task, whose destinations are disjoint, so nothing races.
pub(crate) fn scatter_columns<T: ConvScalar>(
    source: &[T],
    weight: &[T],
    geometry: &ConvGeometry,
) -> Vec<T> {
    let ConvGeometry {
        batch_size: batch,
        in_channels,
        input_height: in_h,
        input_width: in_w,
        out_channels,
        kernel_h,
        kernel_w,
        output_height: out_h,
        output_width: out_w,
        stride,
        padding,
        dilation,
        groups,
    } = *geometry;

    let in_stride = in_channels * in_h * in_w;
    let mut destination = vec![T::default(); batch * in_stride];

    let ohw = out_h * out_w;
    let kh_kw = kernel_h * kernel_w;
    let k_dim = in_channels * kh_kw;
    if ohw == 0 || in_stride == 0 {
        return destination;
    }

    // `k` runs channel-major, so a group owns a contiguous row-block of every
    // `[k_dim, ...]` buffer here and the weight owns one of `[C_out, group_k]`.
    let group_out = out_channels / groups;
    let group_k = (in_channels / groups) * kh_kw;
    let plane = in_h * in_w;

    // A block of images at a time. Taking the whole batch at once meant a
    // `[k_dim, N*OH*OW]` intermediate -- 944MB for a 32x3x224x224 stem -- and
    // taking one image at a time makes the multiply `N` small ones, which for a
    // first layer's shapes runs at a fraction of the rate one large one does.
    // The block is bounded by what the lowering and the copied signal cost, so
    // small images travel in company and large ones alone.
    let span_image = out_channels * ohw;
    let images = images_per_block::<T>(batch, k_dim * ohw, span_image);
    let mut columns = vec![T::default(); images * k_dim * ohw];
    let mut signal = vec![T::default(); images * span_image];

    let mut first = 0;
    while first < batch {
        let last = (first + images).min(batch);
        let span = (last - first) * ohw;
        let block = &mut signal[..out_channels * span];
        block_channel_major::<T>(source, block, first, last, out_channels, ohw);

        for g in 0..groups {
            // SAFETY: group `g` reads `weight` rows `[g*group_out,
            // (g+1)*group_out)` -- stored `[group_out, group_k]` and read
            // transposed by stride, which is why this never materialises the
            // transpose -- against `block` rows `[g*group_out, ..)`, writing
            // `columns` rows `[g*group_k, ..)`. Each is a row-block of a
            // contiguous buffer exactly `groups` times its size.
            unsafe {
                T::gemm_tn(
                    group_k,
                    group_out,
                    span,
                    weight.as_ptr().add(g * group_out * group_k),
                    block.as_ptr().add(g * group_out * span),
                    columns.as_mut_ptr().add(g * group_k * span),
                );
            }
        }

        // Scatter the block's columns back onto its grids, one channel plane
        // per task: a channel's taps write only its own plane of its own image,
        // so the adds never race. Parallelising over the batch alone left a
        // single-image backward pass -- and every batch smaller than the pool
        // -- on one core.
        //
        // Output positions are walked as nested loops with their in-bounds
        // range hoisted, for the reason the forward's im2col does the same:
        // recovering `(oh, ow)` from a flat `p` cost two runtime-divisor
        // divisions per element, over 4.7M elements for a 16x32x32x32
        // convolution.
        let grids = &mut destination[first * in_stride..][..(last - first) * in_stride];
        par_out_chunks(grids, plane, &|start, target| {
            let index = start / plane;
            let image = index / in_channels;
            let channel = index % in_channels;
            for tap in 0..kh_kw {
                let ky_off = (tap / kernel_w) * dilation.0;
                let kx_off = (tap % kernel_w) * dilation.1;
                let (oh_lo, oh_hi) = in_bounds_range(ky_off, padding.0, in_h, stride.0, out_h);
                let (ow_lo, ow_hi) = in_bounds_range(kx_off, padding.1, in_w, stride.1, out_w);
                let row = (channel * kh_kw + tap) * span + image * ohw;
                for oh in oh_lo..oh_hi {
                    let ih = oh * stride.0 + ky_off - padding.0;
                    let dst = ih * in_w;
                    let src = row + oh * out_w;
                    for ow in ow_lo..ow_hi {
                        let iw = ow * stride.1 + kx_off - padding.1;
                        target[dst + iw] += columns[src + ow];
                    }
                }
            }
        });

        first = last;
    }
    destination
}

/// `source @ im2col(image)` -- the gradient a kernel accumulates from a signal
/// and the grid it was gathered from.
///
/// The other half of the pair: a convolution's weight gradient, and a
/// transposed convolution's, which is the same sum with the two operands
/// swapping which one is the image. Shapes follow [`scatter_columns`].
pub(crate) fn column_weight_gradient<T: ConvScalar>(
    image: &[T],
    source: &[T],
    geometry: &ConvGeometry,
) -> Vec<T> {
    let ConvGeometry {
        batch_size: batch,
        in_channels,
        input_height: in_h,
        input_width: in_w,
        out_channels,
        kernel_h,
        kernel_w,
        output_height: out_h,
        output_width: out_w,
        stride,
        padding,
        dilation,
        groups,
    } = *geometry;

    let ohw = out_h * out_w;
    let kh_kw = kernel_h * kernel_w;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let group_k = group_in * kh_kw;

    let mut gradient = vec![T::default(); out_channels * group_k];
    if ohw == 0 || batch == 0 {
        return gradient;
    }

    // This lowering is transposed relative to the forward's -- one row per
    // output position -- so a group's `k` values are a column-block rather than
    // a row-block and cannot be handed to a GEMM by pointer offset. It is built
    // per group, and a block of images at a time: `[block*OH*OW, group_k]`
    // rather than `[N*OH*OW, group_k]`, which for a 32x3x224x224 stem is a few
    // megabytes of scratch instead of 944.
    //
    // A block rather than one image because the multiply is what this costs:
    // a first layer's weight gradient is `[32, OH*OW] @ [OH*OW, 27]`, and 64 of
    // those run at a fraction of the rate one `[32, 64*OH*OW] @ [.., 27]` does.
    //
    // The sum over blocks runs in image order, so the answer does not move with
    // the thread count.
    let span_image = out_channels * ohw;
    let images = images_per_block::<T>(batch, ohw * group_k, span_image);
    let mut columns = vec![T::default(); images * ohw * group_k];
    let mut signal = vec![T::default(); images * span_image];
    let mut partial = vec![T::default(); group_out * group_k];

    let mut first = 0;
    while first < batch {
        let last = (first + images).min(batch);
        let span = (last - first) * ohw;
        let block = &mut signal[..out_channels * span];
        block_channel_major::<T>(source, block, first, last, out_channels, ohw);

        for g in 0..groups {
            // `k` is walked as nested `(ic, ky, kx)` loops rather than
            // decomposed: three divisions per element, over millions of them.
            // The buffer is reused between blocks, so the taps that fall in the
            // padding have to be cleared rather than left as the last block's.
            par_out_chunks(&mut columns[..span * group_k], group_k, &|start, row| {
                row.fill(T::default());
                let position = start / group_k;
                let n = first + position / ohw;
                let p = position % ohw;
                let oh = p / out_w;
                let ow = p % out_w;
                let mut k = 0usize;
                for ic in g * group_in..(g + 1) * group_in {
                    let grid = (n * in_channels + ic) * in_h * in_w;
                    for ky in 0..kernel_h {
                        let ih = oh * stride.0 + ky * dilation.0;
                        let row_ok = ih >= padding.0 && ih < in_h + padding.0;
                        let ih = ih.wrapping_sub(padding.0);
                        for kx in 0..kernel_w {
                            let iw = ow * stride.1 + kx * dilation.1;
                            if row_ok && iw >= padding.1 && iw < in_w + padding.1 {
                                row[k] = image[grid + ih * in_w + (iw - padding.1)];
                            }
                            k += 1;
                        }
                    }
                }
            });
            // SAFETY: `block` rows `[g*group_out, ..)` form a
            // `[group_out, span]` block, `columns` is `[span, group_k]` in full
            // and `partial` is `[group_out, group_k]`. All contiguous
            // row-major.
            unsafe {
                T::gemm(
                    group_out,
                    span,
                    group_k,
                    block.as_ptr().add(g * group_out * span),
                    columns.as_ptr(),
                    partial.as_mut_ptr(),
                );
            }
            let target = &mut gradient[g * group_out * group_k..][..group_out * group_k];
            for (slot, &value) in target.iter_mut().zip(partial.iter()) {
                *slot += value;
            }
        }

        first = last;
    }
    gradient
}

/// Sum a `[N, C, H, W]` signal down to one value per channel -- what a bias
/// gradient is, in both directions.
pub(crate) fn channel_sums<T: ConvScalar>(
    source: &[T],
    batch: usize,
    channels: usize,
    plane: usize,
) -> Vec<T> {
    let mut sums = vec![T::default(); channels];
    par_out_chunks(&mut sums, 1, &|channel, slot| {
        let mut total = T::default();
        for n in 0..batch {
            let base = (n * channels + channel) * plane;
            for offset in 0..plane {
                total += source[base + offset];
            }
        }
        slot[0] = total;
    });
    sums
}

/// Scratch the lowering may hold at once, in bytes.
///
/// Large enough that the GEMM sees a few thousand columns, which is where it
/// runs fastest; small enough that the lowered block is still in cache when the
/// GEMM reads it back.
const COL_BLOCK_BYTES: usize = 8 << 20;

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
        dilation,
        groups,
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
    let kh_kw = kernel_h * kernel_w;
    // Per-group shapes. The lowered `cols` still covers every input channel,
    // and a group's rows within it are contiguous because `k` runs
    // channel-major -- so the groups are `groups` GEMMs over row-blocks of the
    // same three buffers, with no repacking.
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let group_k = group_in * kh_kw;

    let mut output_vec = vec![T::default(); batch_size * out_channels * ohw];

    if !output_vec.is_empty() {
        // Lowering the whole image set at once means `k_dim * n_cols` elements
        // of scratch -- 944MB for a 32x3x224x224 stem -- written once and read
        // straight back out of main memory by the GEMM. A block of output rows
        // is lowered, multiplied and scattered while it is still in cache
        // instead, which bounds the scratch whatever the image is and lands the
        // GEMM near the column count it runs fastest at (below about a thousand
        // columns its efficiency falls off a cliff; above about eight thousand
        // it starts paying for memory again).
        let per_row = (k_dim * output_width).max(1);
        let rows_total = batch_size * output_height;
        let rows_per_block = (COL_BLOCK_BYTES / (per_row * std::mem::size_of::<T>()).max(1))
            .clamp(1, rows_total.max(1));
        let block_n = rows_per_block * output_width;

        let mut cols = vec![T::default(); k_dim * block_n];
        let mut gemm_out = vec![T::default(); out_channels * block_n];

        let mut first_row = 0;
        while first_row < rows_total {
            let last_row = (first_row + rows_per_block).min(rows_total);
            let rows = last_row - first_row;
            let width = rows * output_width;

            // Build the block row by row (one row per kernel-input index `k`),
            // so each row is written contiguously.
            //
            // The output position is walked rather than recovered from a flat
            // counter: decomposing the counter needed four integer divisions
            // per element -- by `ohw` and `output_width`, both runtime values,
            // so they stay real divisions -- across 4.7M elements for a
            // 16x32x32x32 conv.
            //
            // The in-bounds range of output positions is computed once per row
            // instead of testing each element: padding only ever clips a prefix
            // and a suffix. The buffer is reused between blocks, so those have
            // to be cleared rather than merely left alone.
            par_out_chunks(&mut cols[..k_dim * width], width, &|start, row| {
                row.fill(T::default());
                let k = start / width;
                let ic = k / kh_kw;
                let rem = k % kh_kw;
                let ky = rem / kernel_w;
                let kx = rem % kernel_w;
                // The tap's offset into the input is `ky * dilation`, so the
                // in-bounds range is the undilated one evaluated at that offset
                // -- `in_bounds_range` never needed to know about dilation.
                let ky_off = ky * dilation.0;
                let kx_off = kx * dilation.1;
                let (oh_lo, oh_hi) =
                    in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
                let (ow_lo, ow_hi) =
                    in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
                if oh_lo >= oh_hi || ow_lo >= ow_hi {
                    return;
                }
                let span = ow_hi - ow_lo;
                for position in first_row..last_row {
                    let n = position / output_height;
                    let oh = position % output_height;
                    if oh < oh_lo || oh >= oh_hi {
                        continue;
                    }
                    let ih = oh * stride.0 + ky_off - padding.0;
                    let src = ((n * in_channels + ic) * input_height + ih) * input_width;
                    let dst = (position - first_row) * output_width + ow_lo;
                    // Dilation moves where the run starts, not how it is
                    // spaced: consecutive output columns are still consecutive
                    // input columns when the stride is 1, so the contiguous
                    // copy survives.
                    if stride.1 == 1 {
                        let s = src + ow_lo + kx_off - padding.1;
                        row[dst..dst + span].copy_from_slice(&input_data[s..s + span]);
                    } else {
                        for (i, slot) in row[dst..dst + span].iter_mut().enumerate() {
                            let iw = (ow_lo + i) * stride.1 + kx_off - padding.1;
                            *slot = input_data[src + iw];
                        }
                    }
                }
            });

            for g in 0..groups {
                // SAFETY: within group `g`, `weight_data` offset by
                // `g * group_out * group_k` is [group_out, group_k], `cols`
                // offset by `g * group_k * width` is [group_k, width], and
                // `gemm_out` offset by `g * group_out * width` is
                // [group_out, width]. All three are row-blocks of contiguous
                // row-major buffers whose lengths are exactly `groups` times
                // these, so no block runs past its allocation and no two groups
                // overlap.
                unsafe {
                    T::gemm(
                        group_out,
                        group_k,
                        width,
                        weight_data.as_ptr().add(g * group_out * group_k),
                        cols.as_ptr().add(g * group_k * width),
                        gemm_out.as_mut_ptr().add(g * group_out * width),
                    );
                }
            }

            // Scatter [C_out, width] into [N, C_out, oh, ow], adding bias. A
            // block is a run of output rows, so within one image each channel
            // takes a contiguous run of its plane; a block that reaches the end
            // of an image carries on into the next one, which is why this walks
            // images rather than assuming one.
            let mut position = first_row;
            while position < last_row {
                let n = position / output_height;
                let from = position % output_height;
                let to = (last_row - n * output_height).min(output_height);
                let taken = (to - from) * output_width;
                let offset = (position - first_row) * output_width;
                let image = &mut output_vec[n * out_channels * ohw..][..out_channels * ohw];
                par_out_chunks(image, ohw, &|start, plane| {
                    let oc = start / ohw;
                    let source = &gemm_out[oc * width + offset..][..taken];
                    let target = &mut plane[from * output_width..][..taken];
                    for (o, &v) in target.iter_mut().zip(source) {
                        *o = v;
                        if let Some(bd) = bias_data {
                            *o += bd[oc];
                        }
                    }
                });
                position += to - from;
            }

            first_row = last_row;
        }
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
        let out = conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0), (1, 1), 1).unwrap();
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
        let out = conv2d(&input, &weight, None, (2, 2), (1, 1), (1, 1), 1).unwrap();
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
        let result = conv2d(&input, &weight, None, (1, 1), (0, 0), (1, 1), 1);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::*;
    use crate::tensor::Shape;
    use std::sync::Arc;

    fn tensor(values: Vec<f64>, dims: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(values, Device::cpu())),
            Shape::new(dims),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    /// The defining identity: a transposed convolution is the adjoint of a
    /// convolution, so `<conv_transpose(x), y> == <x, conv(y)>` for every `x`
    /// and `y` of the matching shapes. This catches an index mapping that is
    /// wrong in any way at all, without a reference implementation.
    #[test]
    fn transposed_convolution_is_the_adjoint_of_convolution() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        };

        for &(n, cin, cout, h, w, kh, kw, s0, s1, p0, p1, op0, op1, d0, d1, g) in &[
            (
                1usize, 1usize, 1usize, 4usize, 4usize, 3usize, 3usize, 1usize, 1usize, 0usize,
                0usize, 0usize, 0usize, 1usize, 1usize, 1usize,
            ),
            (2, 3, 2, 5, 4, 3, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1),
            (1, 4, 6, 6, 6, 3, 3, 2, 1, 1, 1, 1, 0, 2, 1, 2),
            (2, 4, 4, 3, 5, 2, 3, 3, 2, 0, 1, 2, 1, 1, 2, 4),
        ] {
            let xs: Vec<f64> = (0..n * cin * h * w).map(|_| next()).collect();
            let ws: Vec<f64> = (0..cin * (cout / g) * kh * kw).map(|_| next()).collect();
            let x = tensor(xs, vec![n, cin, h, w]);
            let weight = tensor(ws, vec![cin, cout / g, kh, kw]);

            let up = conv_transpose2d(
                &x,
                &weight,
                None,
                (s0, s1),
                (p0, p1),
                (op0, op1),
                (d0, d1),
                g,
            )
            .unwrap();
            let up_dims = up.shape().dims().to_vec();

            let ys: Vec<f64> = (0..up_dims.iter().product::<usize>())
                .map(|_| next())
                .collect();
            let y = tensor(ys, up_dims.clone());

            // <conv_transpose(x), y>
            let left: f64 = up
                .data()
                .as_f64_slice()
                .unwrap()
                .iter()
                .zip(y.data().as_f64_slice().unwrap())
                .map(|(a, b)| a * b)
                .sum();

            // <x, conv(y)>
            let down = conv2d(&y, &weight, None, (s0, s1), (p0, p1), (d0, d1), g).unwrap();
            assert_eq!(down.shape().dims(), &[n, cin, h, w], "adjoint shape");
            let right: f64 = down
                .data()
                .as_f64_slice()
                .unwrap()
                .iter()
                .zip(x.data().as_f64_slice().unwrap())
                .map(|(a, b)| a * b)
                .sum();

            assert!(
                (left - right).abs() <= 1e-11 * left.abs().max(1.0),
                "adjoint identity failed for n={n} cin={cin} cout={cout} \
                 stride=({s0},{s1}) pad=({p0},{p1}) outpad=({op0},{op1}) \
                 dil=({d0},{d1}) groups={g}: {left} vs {right}"
            );
        }
    }
}
