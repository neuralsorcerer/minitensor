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

mod rounded;
mod rounded_f64;
mod volume;
pub use self::volume::conv3d;
pub(crate) use self::volume::{Volumes, slices_inside, slices_outside};

/// The floating-point types convolution is implemented for.
///
/// Every entry of a convolution, and of each of its gradients, is the float
/// nearest its exact value. A float32 one is rounded from float64 sums of its
/// exact products, which [`rounded`] takes; a float64 one has no wider float
/// to add in, and [`rounded_f64`] puts each of the three as matrix products
/// and dots that are rounded once. What the two share -- the lowering of an
/// input into columns, a block of positions at a time -- is written once
/// over this trait.
pub(crate) trait ConvScalar: Copy + Default + Send + Sync {
    const DTYPE: DataType;
    fn slice(data: &TensorData) -> Option<&[Self]>;
    fn into_tensor_data(values: Vec<Self>, device: Device) -> TensorData;
    /// The convolution of `input` by `weight`, `bias` added to each output
    /// channel; `[N, C_out, OH, OW]` from `[N, C_in, H, W]`.
    fn convolve(
        input: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>>;
    /// `col2im(weight^T @ source)`, `bias` added to each channel of the grid
    /// it writes -- a signal scattered back through a kernel onto the larger
    /// grid it was gathered from.
    ///
    /// This is one operation with two names. Read backwards it is a
    /// convolution's gradient with respect to its input; read forwards it is
    /// a transposed convolution, which is why `conv_transpose2d` is not a
    /// separate algorithm. The two differ only in which side the caller
    /// calls the input.
    ///
    /// In `geometry`, the `input_*` fields describe the grid being written
    /// and the `output_*` fields the signal being read -- the convolution's
    /// own naming, kept so that a caller coming from either direction reads
    /// the same struct. `weight` is `[out_channels, in_channels / groups, kH,
    /// kW]`.
    fn scatter(
        source: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>>;
    /// `source @ im2col(image)^T` -- the gradient a kernel accumulates from a
    /// signal and the grid it was gathered from.
    ///
    /// The other half of the pair: a convolution's weight gradient, and a
    /// transposed convolution's, which is the same sum with the two operands
    /// swapping which one is the image. Shapes follow
    /// [`ConvScalar::scatter`].
    fn weight_gradient(
        image: &[Self],
        source: &[Self],
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>>;
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
    fn convolve(
        input: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        rounded::convolve(input, weight, bias, geometry)
    }
    fn scatter(
        source: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        Ok(rounded::scatter(source, weight, bias, geometry))
    }
    fn weight_gradient(
        image: &[Self],
        source: &[Self],
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        Ok(rounded::weight_gradient(image, source, geometry))
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
    fn convolve(
        input: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        rounded_f64::convolve(input, weight, bias, geometry)
    }
    fn scatter(
        source: &[Self],
        weight: &[Self],
        bias: Option<&[Self]>,
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        rounded_f64::scatter(source, weight, bias, geometry)
    }
    fn weight_gradient(
        image: &[Self],
        source: &[Self],
        geometry: &ConvGeometry,
    ) -> Result<Vec<Self>> {
        rounded_f64::weight_gradient(image, source, geometry)
    }
}

/// The zeros `padding="same"` puts before and after one spatial axis, so a
/// stride-1 convolution keeps that axis's length.
///
/// The kernel's taps span `dilation * (kernel - 1) + 1` positions, so that many
/// less one have to be added in total. When the total is odd the extra zero
/// goes after the axis, which keeps every output centred on the same input
/// position the symmetric case centres it on.
pub fn same_padding(kernel: usize, dilation: usize) -> (usize, usize) {
    let total = dilation * kernel.saturating_sub(1);
    (total / 2, total - total / 2)
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
    // Checked before the 2-D call, whose message would give the kernel as
    // `1x0`.
    if weight_dims[2] == 0 {
        return Err(MinitensorError::invalid_operation(
            "conv1d kernel size must be greater than zero, got 0",
        ));
    }

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

    // A kernel with no taps has no window to slide, and its span below,
    // `dilation * (kernel - 1) + 1`, would wrap around.
    if kernel_h == 0 || kernel_w == 0 {
        return Err(MinitensorError::invalid_operation(format!(
            "kernel size must be greater than zero, got {kernel_h}x{kernel_w}"
        )));
    }
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
        return Err(MinitensorError::shape_mismatch_explained(
            vec![weight_in_channels],
            vec![in_channels / groups],
            format!(
                "conv2d: the input has {in_channels} channels (shape {:?}), but the weight \
                 {:?} with groups={groups} reads {} -- check the input's channel axis, \
                 dimension 1",
                input.shape().dims(),
                weight.shape().dims(),
                weight_in_channels * groups
            ),
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
    // A transposed convolution's output grows with the stride, which the caller
    // sets: stride 10**5 on a 2x2 input asks for terabytes and aborted.
    TensorData::ensure_allocatable(output_shape.try_numel()?, input.dtype())?;

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

    // A kernel with no taps has no window to slide, and its span below,
    // `dilation * (kernel - 1) + 1`, would wrap around.
    if kernel_h == 0 || kernel_w == 0 {
        return Err(MinitensorError::invalid_operation(format!(
            "kernel size must be greater than zero, got {kernel_h}x{kernel_w}"
        )));
    }
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
        return Err(MinitensorError::shape_mismatch_explained(
            vec![in_channels],
            vec![weight.size(0)?],
            format!(
                "conv_transpose2d: the input has {in_channels} channels (shape {:?}), but \
                 the weight {:?} reads {} -- its first axis is the input channels",
                input.shape().dims(),
                weight.shape().dims(),
                weight.size(0)?
            ),
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
    // A transposed convolution's output grows with the stride, which the caller
    // sets: stride 10**5 on a 2x2 input asks for terabytes and aborted.
    TensorData::ensure_allocatable(output_shape.try_numel()?, input.dtype())?;

    macro_rules! scatter {
        ($ty:ty) => {{
            let signal = <$ty as ConvScalar>::slice(contiguous.data()).ok_or_else(|| {
                MinitensorError::internal_error("conv_transpose2d: input dtype does not match")
            })?;
            let kernel = <$ty as ConvScalar>::slice(weight_view.data()).ok_or_else(|| {
                MinitensorError::internal_error("conv_transpose2d: weight dtype does not match")
            })?;
            let bias_view = bias.map(|b| b.contiguous()).transpose()?;
            let added = match &bias_view {
                Some(b) => Some(<$ty as ConvScalar>::slice(b.data()).ok_or_else(|| {
                    MinitensorError::internal_error("conv_transpose2d: bias dtype does not match")
                })?),
                None => None,
            };
            let values = <$ty as ConvScalar>::scatter(signal, kernel, added, &geometry)?;
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
    // Checked before the 2-D call, whose message would give the kernel as
    // `1x0`.
    if weight_dims[2] == 0 {
        return Err(MinitensorError::invalid_operation(
            "conv_transpose1d kernel size must be greater than zero, got 0",
        ));
    }
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
#[derive(Clone, Copy, Debug)]
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

/// Lower output positions `[first, last)` into `cols`, one row per
/// kernel-input index `k` -- the `[K, last - first]` matrix a convolution's
/// GEMM contracts, `K = C_in * kH * kW` running channel-major.
///
/// In `geometry` the `input_*` fields describe the grid being read and the
/// `output_*` fields the positions. Every element of `cols` is written.
fn lower_positions<T: ConvScalar>(
    input_data: &[T],
    cols: &mut [T],
    first: usize,
    last: usize,
    geometry: &ConvGeometry,
) {
    let ConvGeometry {
        in_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        ..
    } = *geometry;
    let width = last - first;
    let kh_kw = kernel_h * kernel_w;
    let ohw = output_height * output_width;
    // Whole rows whenever the block starts and ends on row boundaries: the
    // in-bounds column range is then the same for all of them and each row
    // lands at a fixed offset.
    let row_aligned =
        output_width > 0 && first.is_multiple_of(output_width) && last.is_multiple_of(output_width);
    if width == 0 {
        return;
    }
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
    par_out_chunks(
        &mut cols[..in_channels * kh_kw * width],
        width,
        &|start, row| {
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
            // Dilation moves where a run starts, not how it is spaced:
            // consecutive output columns are still consecutive input
            // columns when the stride is 1, so the contiguous copy
            // survives in both walks below.
            let mut copy = |n: usize, oh: usize, lo: usize, hi: usize, dst: usize| {
                let span = hi - lo;
                let ih = oh * stride.0 + ky_off - padding.0;
                let src = ((n * in_channels + ic) * input_height + ih) * input_width;
                if stride.1 == 1 {
                    let s = src + lo + kx_off - padding.1;
                    row[dst..dst + span].copy_from_slice(&input_data[s..s + span]);
                } else {
                    for (i, slot) in row[dst..dst + span].iter_mut().enumerate() {
                        let iw = (lo + i) * stride.1 + kx_off - padding.1;
                        *slot = input_data[src + iw];
                    }
                }
            };

            if row_aligned {
                // Whole rows, which is every shape one row fits in. The
                // in-bounds column range is then the same for all of them
                // and each row lands at a fixed offset, so neither is
                // recomputed -- this is the walk the row-blocked version
                // did, kept because it is the common one.
                for position in (first..last).step_by(output_width) {
                    let row_index = position / output_width;
                    let oh = row_index % output_height;
                    if oh < oh_lo || oh >= oh_hi {
                        continue;
                    }
                    let n = row_index / output_height;
                    copy(n, oh, ow_lo, ow_hi, (position - first) + ow_lo);
                }
            } else {
                // A row wider than the whole budget, so a block starts and
                // ends part way along one. The position is carried forward
                // rather than recovered from the counter: three divisions
                // for the block instead of two for every row.
                let mut position = first;
                let mut n = first / ohw;
                let mut oh = (first % ohw) / output_width;
                let mut ow = (first % ohw) % output_width;
                while position < last {
                    let take = (output_width - ow).min(last - position);
                    if oh >= oh_lo && oh < oh_hi {
                        let lo = ow.max(ow_lo);
                        let hi = (ow + take).min(ow_hi);
                        if lo < hi {
                            copy(n, oh, lo, hi, (position - first) + (lo - ow));
                        }
                    }
                    position += take;
                    ow += take;
                    if ow == output_width {
                        ow = 0;
                        oh += 1;
                        if oh == output_height {
                            oh = 0;
                            n += 1;
                        }
                    }
                }
            }
        },
    );
}

/// The most taps a direct convolution takes for an output channel, `3 * 7 *
/// 7`: a stem over an image's three channels.
const DIRECT_MAX_TAPS: usize = 147;

/// Whether a float32 convolution goes straight from the input -- its output
/// and both of its gradients, as [`rounded`] takes them -- rather than through
/// the lowered columns and a GEMM.
///
/// The lowering writes every tap of every position once, and pays that back
/// only when the multiply reuses it across many output channels. A group with
/// few of them -- a depthwise convolution has one -- is a GEMM with a handful
/// of rows, and was the slowest thing in a network: a 3x3 depthwise layer over
/// `[16, 32, 56, 56]` took 29.2ms forward and 55.9ms with its backward, and
/// takes 1.2 and 4.8 straight from the input. A first layer reading one or
/// three channels is the same shape of problem. Measured forward and backward,
/// four threads, float32:
///
/// ```text
///   conv                       lowered    direct
///   depthwise 3x3, 32 ch        55.9ms     4.8ms
///   depthwise 7x7, 96 ch       146.0      25.0
///   64 -> 1, 3x3                14.5       2.4
///   64 -> 4, 3x3                19.4      10.0
///   3 -> 8, 3x3                  5.0       2.1
///   1 -> 16, 3x3                 4.8       3.5
///   64 -> 64, 3x3, 8 groups     22.2      16.4
///   ---- kept lowered ----
///   64 -> 4, 3x3, stride 2       6.4       8.5
///   64 -> 8, 3x3                13.3      15.7
///   32 -> 8, 3x3                 8.1       8.2
/// ```
///
/// Each output costs the direct path all of its taps, where the GEMM shares
/// the lowering between output channels, so with more than four of them it
/// pays only for a small kernel, and with more than eight only for one as
/// small as a 3x3 over one channel into sixteen. It also needs rows: a stride
/// along them makes its inner loop a gather instead of a contiguous
/// multiply-add, and a short row is mostly loop overhead.
fn direct_pays(geometry: &ConvGeometry) -> bool {
    let group_out = geometry.out_channels / geometry.groups;
    let group_k = (geometry.in_channels / geometry.groups) * geometry.kernel_h * geometry.kernel_w;
    let contiguous = geometry.stride.1 == 1;
    (group_out <= 4 && (contiguous || group_k <= DIRECT_MAX_TAPS))
        || (group_out <= 8
            && contiguous
            && geometry.output_width >= 14
            && group_k <= DIRECT_MAX_TAPS)
        || (contiguous && group_out * group_k <= 144)
}

/// Scratch the lowering may hold at once, in bytes.
///
/// Large enough that the GEMM sees a few thousand columns, which is where it
/// runs fastest; small enough that the lowered block is still in cache when the
/// GEMM reads it back.
const COL_BLOCK_BYTES: usize = 8 << 20;

/// The forward pass, for one element type: the operands' storage handed to
/// [`ConvScalar::convolve`].
fn conv2d_forward<T: ConvScalar>(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    geom: ConvGeometry,
) -> Result<TensorData> {
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
    let output = T::convolve(input_data, weight_data, bias_data, &geom)?;
    Ok(T::into_tensor_data(output, input.device()))
}

/// What multiplies a block of the lowered columns by the kernel: given the
/// block's `width`, its `[K, width]` columns, and the `[C_out, width]`
/// products to fill.
type BlockProducts<'a, T> = dyn FnMut(usize, &[T], &mut [T]) -> Result<()> + 'a;

/// The convolution through its lowered columns: each block of output
/// positions lowered into a `[K, width]` matrix, `K = C_in * kH * kW` running
/// channel-major, which `products` multiplies by the kernel into `[C_out,
/// width]`, and that placed into the `[N, C_out, OH, OW]` output.
///
/// The kernel is already laid out as `[C_out, K / groups]`, and a group's
/// rows of the lowered matrix are contiguous because `k` runs channel-major
/// -- so the groups are `groups` products over row-blocks of the same three
/// buffers, with no repacking.
///
/// Lowering the whole image set at once means `K * N * OH * OW` elements of
/// scratch -- 944MB for a 32x3x224x224 stem -- written once and read straight
/// back out of main memory. A block is lowered, multiplied and placed while
/// it is still in cache instead, which bounds the scratch whatever the image
/// is.
///
/// The block is a run of output *positions*, not of output rows. Rows bound
/// the scratch only while one row fits in it: a single row costs `K *
/// output_width`, so a 1-D convolution -- one row, a million columns -- asked
/// for that whole product however small the budget was. With a million-tap
/// kernel that is 17.6TB, and the allocation failure aborts the process
/// rather than raising. Whole rows are taken whenever one fits, so that each
/// copy starts on a row boundary; only a row too wide for the budget gets a
/// block that starts or ends part way along one.
fn convolve_lowered<T: ConvScalar>(
    input: &[T],
    geometry: &ConvGeometry,
    products: &mut BlockProducts<'_, T>,
) -> Result<Vec<T>> {
    let ConvGeometry {
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        ..
    } = *geometry;
    let ohw = output_height * output_width;
    let k_dim = in_channels * kernel_h * kernel_w;
    let mut output = vec![T::default(); batch_size * out_channels * ohw];
    if output.is_empty() {
        return Ok(output);
    }
    let per_column = (k_dim * std::mem::size_of::<T>()).max(1);
    let positions_total = batch_size * ohw;
    let budget = (COL_BLOCK_BYTES / per_column).clamp(1, positions_total);
    let block_n = if budget >= output_width {
        (budget / output_width) * output_width
    } else {
        budget
    };
    let mut cols = vec![T::default(); k_dim * block_n];
    let mut block = vec![T::default(); out_channels * block_n];

    let mut first = 0;
    while first < positions_total {
        let last = (first + block_n).min(positions_total);
        let width = last - first;
        lower_positions::<T>(input, &mut cols, first, last, geometry);
        products(
            width,
            &cols[..k_dim * width],
            &mut block[..out_channels * width],
        )?;

        // `[C_out, width]` into `[N, C_out, oh, ow]`. A block is a run of
        // output positions, so within one image each channel takes a
        // contiguous run of its plane, and a block that reaches the end of an
        // image carries on into the next. Every plane the block touches is
        // one task of a single dispatch: dispatched image by image, a batch
        // of eight 30x30 images paid the pool's round trip eight times.
        let first_image = first / ohw;
        let last_image = (last - 1) / ohw + 1;
        let region = &mut output[first_image * out_channels * ohw..last_image * out_channels * ohw];
        let block = &block;
        par_out_chunks(region, ohw, &|start, plane| {
            let index = start / ohw;
            let (n, oc) = (first_image + index / out_channels, index % out_channels);
            let image_start = n * ohw;
            let from = first.max(image_start) - image_start;
            let to = last.min(image_start + ohw) - image_start;
            plane[from..to]
                .copy_from_slice(&block[oc * width + (image_start + from - first)..][..to - from]);
        });
        first = last;
    }
    Ok(output)
}

/// The terms of entry `k` of output channel `oc`'s kernel slice in a weight
/// gradient: each value of the signal's channel beside the value of the
/// image its tap reads for it, over every image.
fn weight_terms<T: Copy>(
    oc: usize,
    k: usize,
    image: &[T],
    source: &[T],
    geometry: &ConvGeometry,
) -> (Vec<T>, Vec<T>) {
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
    } = *geometry;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let kh_kw = kernel_h * kernel_w;
    let (local, tap) = (k / kh_kw, k % kh_kw);
    let ky_off = (tap / kernel_w) * dilation.0;
    let kx_off = (tap % kernel_w) * dilation.1;
    let ic = (oc / group_out) * group_in + local;
    let (oh_lo, oh_hi) = in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
    let (ow_lo, ow_hi) = in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
    let span = ow_hi - ow_lo;
    let (mut signal, mut reads) = (Vec::new(), Vec::new());
    for n in 0..batch_size {
        for oh in oh_lo..oh_hi {
            let ih = oh * stride.0 + ky_off - padding.0;
            let from = ((n * out_channels + oc) * output_height + oh) * output_width + ow_lo;
            signal.extend_from_slice(&source[from..from + span]);
            let row =
                &image[((n * in_channels + ic) * input_height + ih) * input_width..][..input_width];
            reads.extend((ow_lo..ow_hi).map(|ow| row[ow * stride.1 + kx_off - padding.1]));
        }
    }
    (signal, reads)
}

/// What the tests of the rounded convolutions share: a geometry from a
/// convolution's arguments, and every term of it from the definition.
#[cfg(test)]
pub(crate) mod testing {
    use super::ConvGeometry;

    pub(crate) fn geometry(
        (batch, channels, height, width): (usize, usize, usize, usize),
        (outputs, kernel): (usize, (usize, usize)),
        (stride, padding, dilation): ((usize, usize), (usize, usize), (usize, usize)),
        groups: usize,
    ) -> ConvGeometry {
        let span = |k: usize, d: usize| d * (k - 1) + 1;
        ConvGeometry {
            batch_size: batch,
            in_channels: channels,
            input_height: height,
            input_width: width,
            out_channels: outputs,
            kernel_h: kernel.0,
            kernel_w: kernel.1,
            output_height: (height + 2 * padding.0 - span(kernel.0, dilation.0)) / stride.0 + 1,
            output_width: (width + 2 * padding.1 - span(kernel.1, dilation.1)) / stride.1 + 1,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Where tap `k` of output `o` reads the input, written from the
    /// definition rather than from the kernels' ranges.
    fn read_at(
        o: usize,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        extent: usize,
    ) -> Option<usize> {
        let at = (o * stride + k * dilation) as isize - padding as isize;
        (0..extent as isize).contains(&at).then_some(at as usize)
    }

    /// Every term of the convolution, as `(output, input, tap)` indices.
    pub(crate) fn terms(g: &ConvGeometry) -> Vec<(usize, usize, usize)> {
        let group_in = g.in_channels / g.groups;
        let group_out = g.out_channels / g.groups;
        let mut all = Vec::new();
        for n in 0..g.batch_size {
            for oc in 0..g.out_channels {
                for oh in 0..g.output_height {
                    for ow in 0..g.output_width {
                        for local in 0..group_in {
                            let ic = (oc / group_out) * group_in + local;
                            for ky in 0..g.kernel_h {
                                for kx in 0..g.kernel_w {
                                    let (Some(ih), Some(iw)) = (
                                        read_at(
                                            oh,
                                            ky,
                                            g.stride.0,
                                            g.padding.0,
                                            g.dilation.0,
                                            g.input_height,
                                        ),
                                        read_at(
                                            ow,
                                            kx,
                                            g.stride.1,
                                            g.padding.1,
                                            g.dilation.1,
                                            g.input_width,
                                        ),
                                    ) else {
                                        continue;
                                    };
                                    let output = ((n * g.out_channels + oc) * g.output_height + oh)
                                        * g.output_width
                                        + ow;
                                    let input = ((n * g.in_channels + ic) * g.input_height + ih)
                                        * g.input_width
                                        + iw;
                                    let tap = ((oc * group_in + local) * g.kernel_h + ky)
                                        * g.kernel_w
                                        + kx;
                                    all.push((output, input, tap));
                                }
                            }
                        }
                    }
                }
            }
        }
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor, TensorData},
    };

    /// The lowered matrix is blocked by output position, not by output row,
    /// and this is the case that distinguishes the two: a row so wide that the
    /// whole budget does not hold one. The answer has to be the same one the
    /// row-blocked walk gives, which is what the reference loop below checks
    /// element by element.
    #[test]
    fn a_row_wider_than_the_block_budget_still_convolves() {
        let width = 4096usize;
        let taps = 512usize;
        // `k_dim * output_width` is 512 * 3585 here -- 14MB of scratch at
        // f32, against the 8MB budget, so the block cannot be a whole row.
        let signal: Vec<f32> = (0..width).map(|i| ((i % 17) as f32) - 8.0).collect();
        let kernel: Vec<f32> = (0..taps).map(|i| ((i % 5) as f32) * 0.25).collect();

        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(signal.clone(), Device::cpu())),
            Shape::new(vec![1, 1, 1, width]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let weight = Tensor::new(
            Arc::new(TensorData::from_vec_f32(kernel.clone(), Device::cpu())),
            Shape::new(vec![1, 1, 1, taps]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let out = conv2d(&input, &weight, None, (1, 1), (0, 0), (1, 1), 1).unwrap();
        let got = out.data().as_f32_slice().unwrap();
        assert_eq!(got.len(), width - taps + 1);

        for (position, &value) in got.iter().enumerate() {
            let want: f32 = (0..taps).map(|k| signal[position + k] * kernel[k]).sum();
            assert!(
                (value - want).abs() <= 1e-3 * want.abs().max(1.0),
                "position {position}: {value} against {want}"
            );
        }
    }

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
