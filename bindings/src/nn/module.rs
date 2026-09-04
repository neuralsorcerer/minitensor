// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

// `layers` hosts the PyClass wrappers and the module registration function.
// It is a child of this module so its `impl PyReLU`/`impl PyDenseLayer`
// blocks and its `wrap_pyfunction!` calls can reach the pyclass structs and
// `#[pyfunction]`s defined here.
#[path = "init.rs"]
pub mod init;
#[path = "layers.rs"]
mod layers;
pub use self::layers::*;

use crate::device::{PyDevice, resolve_device};
use crate::dtype;
use crate::error::_convert_error;
use crate::serialization::PyStateDict;
use crate::tensor::PyTensor;
use engine::nn::{
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, DenseLayer, FocalLoss, HuberLoss, Layer,
    LogCoshLoss, MAELoss, MSELoss, ReLU, Sequential, Sigmoid, SmoothL1Loss, Softmax, Tanh,
    activation::{ELU, GELU, LeakyReLU},
    attention::MultiheadAttention,
    conv::{Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d},
    dropout::{Dropout, Dropout2d},
    embedding::Embedding,
    normalization::{BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm},
    pooling::{
        AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
        AvgPool2d, MaxPool1d, MaxPool2d, Upsample,
    },
    recurrent::{CellKind, Recurrent},
    utils::{LayerUtils, SequentialUtils},
};
use engine::ops::batch_norm as batch_norm_op;
use engine::ops::conv_transpose1d as conv_transpose1d_op;
use engine::ops::conv_transpose2d as conv_transpose2d_op;
use engine::ops::conv1d as conv1d_op;
use engine::ops::conv2d as conv2d_op;
use engine::ops::grid_sample::{Padding as GridPadding, SampleMode, grid_sample as grid_sample_op};
use engine::ops::interpolate::{InterpolateMode, interpolate as interpolate_op};
use engine::ops::loss::cross_entropy as cross_entropy_op;
use engine::ops::loss::{
    cosine_embedding_loss as cosine_embedding_loss_op, ctc_loss as ctc_loss_op,
    focal_loss as focal_loss_op, hinge_embedding_loss as hinge_embedding_loss_op,
    huber_loss as huber_loss_op, kl_div_loss as kl_div_loss_op, mae_loss as mae_loss_op,
    margin_ranking_loss as margin_ranking_loss_op, poisson_nll_loss as poisson_nll_loss_op,
    smooth_l1_loss as smooth_l1_loss_op, soft_margin_loss as soft_margin_loss_op,
    triplet_margin_loss as triplet_margin_loss_op,
};
use engine::ops::pooling::{
    adaptive_avg_pool1d as adaptive_avg_pool1d_op, adaptive_avg_pool2d as adaptive_avg_pool2d_op,
    adaptive_max_pool1d as adaptive_max_pool1d_op, adaptive_max_pool2d as adaptive_max_pool2d_op,
    avg_pool1d as avg_pool1d_op, avg_pool2d as avg_pool2d_op, max_pool1d as max_pool1d_op,
    max_pool1d_with_indices as max_pool1d_with_indices_op, max_pool2d as max_pool2d_op,
    max_pool2d_with_indices as max_pool2d_with_indices_op,
};
use engine::serialization::{ModelMetadata, ModelSerializer, SerializationFormat, SerializedModel};
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule as Pyo3Module, PyTuple};

fn borrow_tensor<'py>(value: &'py Bound<'py, PyAny>) -> PyResult<PyRef<'py, PyTensor>> {
    if let Ok(tensor) = value.extract::<PyRef<PyTensor>>() {
        return Ok(tensor);
    }

    let py = value.py();
    let inner = value
        .getattr(intern!(py, "_tensor"))
        .map_err(|_| PyTypeError::new_err("expected a minitensor Tensor or core Tensor"))?;
    Ok(inner.extract::<PyRef<PyTensor>>()?)
}

fn borrow_optional_tensor<'py>(
    value: Option<&'py Bound<'py, PyAny>>,
) -> PyResult<Option<PyRef<'py, PyTensor>>> {
    value.map(borrow_tensor).transpose()
}

fn borrow_tensor_mut<'py>(value: &'py Bound<'py, PyAny>) -> PyResult<PyRefMut<'py, PyTensor>> {
    if let Ok(tensor) = value.extract::<PyRefMut<PyTensor>>() {
        return Ok(tensor);
    }

    let py = value.py();
    let inner = value
        .getattr(intern!(py, "_tensor"))
        .map_err(|_| PyTypeError::new_err("expected a minitensor Tensor or core Tensor"))?;
    Ok(inner.extract::<PyRefMut<PyTensor>>()?)
}

fn borrow_optional_tensor_mut<'py>(
    value: Option<&'py Bound<'py, PyAny>>,
) -> PyResult<Option<PyRefMut<'py, PyTensor>>> {
    value.map(borrow_tensor_mut).transpose()
}

/// Affine map `input @ weight.T + bias`, with the weight stored `[out_features, in_features]`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
fn dense_layer(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;

    if weight_tensor.tensor().ndim() != 2 {
        return Err(PyValueError::new_err("weight tensor must be 2-dimensional"));
    }

    let bias_tensor = borrow_optional_tensor(bias)?;
    let output = engine::ops::linalg::linear(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
    )
    .map_err(_convert_error)?;

    Ok(PyTensor::from_tensor(output))
}

fn parse_pair_arg(
    name: &str,
    value: Option<&Bound<PyAny>>,
    default: (usize, usize),
) -> PyResult<(usize, usize)> {
    match value {
        None => Ok(default),
        Some(bound) => {
            if let Ok(scalar) = bound.extract::<isize>() {
                if scalar < 0 {
                    return Err(PyValueError::new_err(format!(
                        "{name} must be non-negative"
                    )));
                }
                let scalar = scalar as usize;
                return Ok((scalar, scalar));
            }

            if let Ok(pair) = bound.extract::<(isize, isize)>() {
                if pair.0 < 0 || pair.1 < 0 {
                    return Err(PyValueError::new_err(format!(
                        "{name} values must be non-negative"
                    )));
                }
                return Ok((pair.0 as usize, pair.1 as usize));
            }

            let seq = bound.extract::<Vec<isize>>()?;
            if seq.len() != 2 {
                return Err(PyTypeError::new_err(format!(
                    "{name} must be an int or a sequence of length 2"
                )));
            }
            if seq[0] < 0 || seq[1] < 0 {
                return Err(PyValueError::new_err(format!(
                    "{name} values must be non-negative"
                )));
            }
            Ok((seq[0] as usize, seq[1] as usize))
        }
    }
}

/// 2-D cross-correlation of `input` with `weight`. `dilation` spaces the kernel taps apart; `groups` splits the channels into that many independent convolutions, so `groups=in_channels` is a depthwise convolution.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1))]
fn conv2d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    dilation: Option<&Bound<PyAny>>,
    groups: usize,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let stride = parse_pair_arg("stride", stride, (1, 1))?;
    let padding = parse_pair_arg("padding", padding, (0, 0))?;
    let dilation = parse_pair_arg("dilation", dilation, (1, 1))?;
    let result = conv2d_op(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
        stride,
        padding,
        dilation,
        groups,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 2-D transposed convolution: scatters each input position across a neighbourhood, growing the grid where `conv2d` shrinks it. `weight` is `[C_in, C_out // groups, kH, kW]` -- input channels first. `output_padding` picks among the input sizes that convolve to the same output size, and must be smaller than `stride`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, output_padding=None, dilation=None, groups=1))]
#[allow(clippy::too_many_arguments)]
fn conv_transpose2d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    output_padding: Option<&Bound<PyAny>>,
    dilation: Option<&Bound<PyAny>>,
    groups: usize,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let stride = parse_pair_arg("stride", stride, (1, 1))?;
    let padding = parse_pair_arg("padding", padding, (0, 0))?;
    let output_padding = parse_pair_arg("output_padding", output_padding, (0, 0))?;
    let dilation = parse_pair_arg("dilation", dilation, (1, 1))?;
    let result = conv_transpose2d_op(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 1-D transposed convolution. See `conv_transpose2d`; `weight` is `[C_in, C_out // groups, K]`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1))]
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let result = conv_transpose1d_op(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 1-D cross-correlation of `input` with `weight`. See `conv2d` for `dilation` and `groups`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
fn conv1d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let result = conv1d_op(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
        stride,
        padding,
        dilation,
        groups,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Largest value in each window along the last dimension. Stride defaults to the window, unlike convolution. With `return_indices` the result is `(values, indices)`, where each index is the position along the axis -- what `max_unpool1d` scatters back into.
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=0, return_indices=false))]
fn max_pool1d(
    py: Python<'_>,
    input: &Bound<PyAny>,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    return_indices: bool,
) -> PyResult<Py<PyAny>> {
    let input_tensor = borrow_tensor(input)?;
    // Pooling defaults its stride to the window, unlike convolution.
    let stride = stride.unwrap_or(kernel_size);
    if !return_indices {
        let result = max_pool1d_op(input_tensor.tensor(), kernel_size, stride, padding)
            .map_err(_convert_error)?;
        return Ok(PyTensor::from_tensor(result).into_pyobject(py)?.into());
    }
    let (values, indices) =
        max_pool1d_with_indices_op(input_tensor.tensor(), kernel_size, stride, padding)
            .map_err(_convert_error)?;
    Ok((
        PyTensor::from_tensor(values),
        PyTensor::from_tensor(indices),
    )
        .into_pyobject(py)?
        .into())
}

/// Mean of each window along the last dimension. Stride defaults to the window.
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=0, count_include_pad=true))]
fn avg_pool1d(
    input: &Bound<PyAny>,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let stride = stride.unwrap_or(kernel_size);
    let result = avg_pool1d_op(
        input_tensor.tensor(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Largest value in each 2-D window. Stride defaults to the window, unlike convolution. With `return_indices` the result is `(values, indices)`, where each index is a flat offset into the unpadded input plane -- what `max_unpool2d` scatters back into.
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None, return_indices=false))]
fn max_pool2d(
    py: Python<'_>,
    input: &Bound<PyAny>,
    kernel_size: &Bound<PyAny>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    return_indices: bool,
) -> PyResult<Py<PyAny>> {
    let input_tensor = borrow_tensor(input)?;
    let kernel = parse_pair_arg("kernel_size", Some(kernel_size), (1, 1))?;
    // Pooling defaults its stride to the window, unlike convolution.
    let stride = parse_pair_arg("stride", stride, kernel)?;
    let padding = parse_pair_arg("padding", padding, (0, 0))?;
    if !return_indices {
        let result = max_pool2d_op(input_tensor.tensor(), kernel, stride, padding)
            .map_err(_convert_error)?;
        return Ok(PyTensor::from_tensor(result).into_pyobject(py)?.into());
    }
    let (values, indices) =
        max_pool2d_with_indices_op(input_tensor.tensor(), kernel, stride, padding)
            .map_err(_convert_error)?;
    Ok((
        PyTensor::from_tensor(values),
        PyTensor::from_tensor(indices),
    )
        .into_pyobject(py)?
        .into())
}

/// Resample a `[N, C, L]` or `[N, C, H, W]` signal to a different size, without parameters. Give exactly one of `size` and `scale_factor`. `mode` is `"nearest"`, or `"linear"`/`"bilinear"` for a weighted average of neighbours. `align_corners` puts the first and last output positions exactly on the first and last input samples; the default spaces them as cell centres instead, which is what makes resampling twice by two match resampling once by four.
#[pyfunction]
#[pyo3(signature = (input, size=None, scale_factor=None, mode="nearest", align_corners=false))]
fn interpolate(
    input: &Bound<PyAny>,
    size: Option<&Bound<PyAny>>,
    scale_factor: Option<&Bound<PyAny>>,
    mode: &str,
    align_corners: bool,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let spatial = input_tensor.tensor().ndim().saturating_sub(2);
    let parsed_mode = InterpolateMode::from_name(mode).map_err(_convert_error)?;

    let sizes = match size {
        Some(value) => Some(parse_spatial_usize("size", value, spatial)?),
        None => None,
    };
    let factors = match scale_factor {
        Some(value) => Some(parse_spatial_f64("scale_factor", value, spatial)?),
        None => None,
    };

    let result = interpolate_op(
        input_tensor.tensor(),
        sizes.as_deref(),
        factors.as_deref(),
        parsed_mode,
        align_corners,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Accept a scalar (broadcast to every spatial axis) or a sequence of exactly that many.
fn parse_spatial_usize(name: &str, value: &Bound<PyAny>, spatial: usize) -> PyResult<Vec<usize>> {
    if let Ok(scalar) = value.extract::<usize>() {
        return Ok(vec![scalar; spatial]);
    }
    let seq = value.extract::<Vec<usize>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be an int or a sequence of ints"))
    })?;
    Ok(seq)
}

/// [`parse_spatial_usize`] for a scale factor, which may be fractional.
fn parse_spatial_f64(name: &str, value: &Bound<PyAny>, spatial: usize) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(vec![scalar; spatial]);
    }
    let seq = value.extract::<Vec<f64>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a number or a sequence of numbers"))
    })?;
    Ok(seq)
}

/// Average pooling to a fixed `output_size`, whatever the input's spatial size is. Windows come from the ratio of the extents, so they can overlap and vary in size; `output_size=1` is the global average pool that ends most convolutional networks.
#[pyfunction]
#[pyo3(signature = (input, output_size))]
fn adaptive_avg_pool2d(input: &Bound<PyAny>, output_size: &Bound<PyAny>) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let size = parse_pair_arg("output_size", Some(output_size), (1, 1))?;
    let result = adaptive_avg_pool2d_op(input_tensor.tensor(), size).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Max pooling to a fixed `output_size`. See `adaptive_avg_pool2d`.
#[pyfunction]
#[pyo3(signature = (input, output_size))]
fn adaptive_max_pool2d(input: &Bound<PyAny>, output_size: &Bound<PyAny>) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let size = parse_pair_arg("output_size", Some(output_size), (1, 1))?;
    let result = adaptive_max_pool2d_op(input_tensor.tensor(), size).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 1-D adaptive average pooling over `[N, C, L]`.
#[pyfunction]
#[pyo3(signature = (input, output_size))]
fn adaptive_avg_pool1d(input: &Bound<PyAny>, output_size: usize) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let result =
        adaptive_avg_pool1d_op(input_tensor.tensor(), output_size).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 1-D adaptive max pooling over `[N, C, L]`.
#[pyfunction]
#[pyo3(signature = (input, output_size))]
fn adaptive_max_pool1d(input: &Bound<PyAny>, output_size: usize) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let result =
        adaptive_max_pool1d_op(input_tensor.tensor(), output_size).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Mean of each 2-D window. Stride defaults to the window.
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None, count_include_pad=true))]
fn avg_pool2d(
    input: &Bound<PyAny>,
    kernel_size: &Bound<PyAny>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let kernel = parse_pair_arg("kernel_size", Some(kernel_size), (1, 1))?;
    let stride = parse_pair_arg("stride", stride, kernel)?;
    let padding = parse_pair_arg("padding", padding, (0, 0))?;
    let result = avg_pool2d_op(
        input_tensor.tensor(),
        kernel,
        stride,
        padding,
        count_include_pad,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Normalize each channel over the batch. In training mode it uses the batch statistics and updates the running ones; in evaluation mode it uses the running ones.
#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=true, momentum=0.1, eps=1e-5))]
#[allow(clippy::too_many_arguments)]
fn batch_norm(
    input: &Bound<PyAny>,
    running_mean: Option<&Bound<PyAny>>,
    running_var: Option<&Bound<PyAny>>,
    weight: Option<&Bound<PyAny>>,
    bias: Option<&Bound<PyAny>>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let mut running_mean_tensor = borrow_optional_tensor_mut(running_mean)?;
    let mut running_var_tensor = borrow_optional_tensor_mut(running_var)?;
    let weight_tensor = borrow_optional_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;

    let rm_tensor = running_mean_tensor.as_mut().map(|t| t.tensor_mut());
    let rv_tensor = running_var_tensor.as_mut().map(|t| t.tensor_mut());
    let result = batch_norm_op(
        input_tensor.tensor(),
        rm_tensor,
        rv_tensor,
        weight_tensor.as_ref().map(|w| w.tensor()),
        bias_tensor.as_ref().map(|b| b.tensor()),
        training,
        momentum,
        eps,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Softmax cross-entropy. The target is either one class index per prediction, or a full score per class. `dim` selects the class axis and defaults to 1.
#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean", dim=1))]
fn cross_entropy(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
    dim: isize,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;

    let axis = engine::ops::normalize_dim_named(
        dim,
        input_tensor.tensor().ndim(),
        "cross_entropy: dim (the class axis)",
    )
    .map_err(_convert_error)?;
    let result = cross_entropy_op(
        input_tensor.tensor(),
        target_tensor.tensor(),
        reduction,
        axis,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Zero each element independently with probability `p` and rescale the rest by `1/(1-p)`, so the mean is unchanged. A no-op when `training` is false.
#[pyfunction(name = "dropout")]
#[pyo3(signature = (input, p=0.5, training=true))]
fn dropout_functional(input: &Bound<PyAny>, p: f64, training: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let mut layer = Dropout::new(Some(p)).map_err(_convert_error)?;
    if training {
        layer.train();
    } else {
        layer.eval();
    }
    let result = layer.forward(tensor.tensor()).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Like `dropout`, but zeroes whole channels rather than individual elements.
#[pyfunction(name = "dropout2d")]
#[pyo3(signature = (input, p=0.5, training=true))]
fn dropout2d_functional(input: &Bound<PyAny>, p: f64, training: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let mut layer = Dropout2d::new(Some(p)).map_err(_convert_error)?;
    if training {
        layer.train();
    } else {
        layer.eval();
    }
    let result = layer.forward(tensor.tensor()).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Mean squared error between predictions and targets.
#[pyfunction(name = "mse_loss")]
#[pyo3(signature = (input, target, reduction=None))]
fn mse_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let loss = MSELoss::new(reduction);
    let result = loss
        .forward(prediction.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `beta` was previously fixed at 1.0 -- `SmoothL1Loss` had no field for it. The default is unchanged.
#[pyfunction(name = "smooth_l1_loss")]
#[pyo3(signature = (input, target, reduction=None, beta=1.0))]
fn smooth_l1_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
    beta: f64,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    // The op validates beta; huber and smooth-l1 differ by a factor of beta and
    // coincide only at 1.0, so this must not just forward to `huber_loss_op`.
    let result = smooth_l1_loss_op(prediction.tensor(), target_tensor.tensor(), beta, reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Squared error within `delta` of the target and linear beyond it, so outliers pull less than under `mse_loss`.
#[pyfunction(name = "huber_loss")]
#[pyo3(signature = (input, target, reduction=None, delta=1.0))]
fn huber_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
    delta: f64,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let result = huber_loss_op(
        prediction.tensor(),
        target_tensor.tensor(),
        delta,
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Mean absolute error between predictions and targets.
#[pyfunction(name = "l1_loss")]
#[pyo3(signature = (input, target, reduction=None))]
fn l1_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let result = mae_loss_op(prediction.tensor(), target_tensor.tensor(), reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Kullback-Leibler divergence from `target` to `input`, both given as probabilities -- *not* as log-probabilities, which is what PyTorch's `kl_div` takes. A zero in `target` contributes nothing, as the definition requires.
#[pyfunction(name = "kl_div")]
#[pyo3(signature = (input, target, reduction=None))]
fn kl_div_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let result = kl_div_loss_op(prediction.tensor(), target_tensor.tensor(), reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Cross-entropy down-weighted on well-classified examples by `(1 - p) ** gamma`, for imbalanced classes.
#[pyfunction(name = "focal_loss")]
#[pyo3(signature = (input, target, alpha=0.25, gamma=2.0, reduction=None))]
fn focal_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    alpha: f64,
    gamma: f64,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let result = focal_loss_op(
        prediction.tensor(),
        target_tensor.tensor(),
        alpha,
        gamma,
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `max(0, -target * (input1 - input2) + margin)`, for a `target` of `+1` where `input1` should rank higher and `-1` where `input2` should.
#[pyfunction(name = "margin_ranking_loss")]
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction=None))]
fn margin_ranking_loss_functional(
    input1: &Bound<PyAny>,
    input2: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let left = borrow_tensor(input1)?;
    let right = borrow_tensor(input2)?;
    let labels = borrow_tensor(target)?;
    let result = margin_ranking_loss_op(
        left.tensor(),
        right.tensor(),
        labels.tensor(),
        margin,
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// The distance itself where `target` is `+1`, and `max(0, margin - distance)` where it is `-1`.
#[pyfunction(name = "hinge_embedding_loss")]
#[pyo3(signature = (input, target, margin=1.0, reduction=None))]
fn hinge_embedding_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let distances = borrow_tensor(input)?;
    let labels = borrow_tensor(target)?;
    let result = hinge_embedding_loss_op(
        distances.tensor(),
        labels.tensor(),
        margin,
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `1 - cos(x1, x2)` where `target` is `+1`, and `max(0, cos(x1, x2) - margin)` where it is `-1`.
#[pyfunction(name = "cosine_embedding_loss")]
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction=None))]
fn cosine_embedding_loss_functional(
    input1: &Bound<PyAny>,
    input2: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let left = borrow_tensor(input1)?;
    let right = borrow_tensor(input2)?;
    let labels = borrow_tensor(target)?;
    let result = cosine_embedding_loss_op(
        left.tensor(),
        right.tensor(),
        labels.tensor(),
        margin,
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `max(0, d(anchor, positive) - d(anchor, negative) + margin)`. With `swap`, the negative distance is the smaller of `d(anchor, negative)` and `d(positive, negative)`, so a triplet whose positive sits closest to the negative still counts as a violation.
#[pyfunction(name = "triplet_margin_loss")]
#[pyo3(signature = (anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=false, reduction=None))]
#[allow(clippy::too_many_arguments)]
fn triplet_margin_loss_functional(
    anchor: &Bound<PyAny>,
    positive: &Bound<PyAny>,
    negative: &Bound<PyAny>,
    margin: f64,
    p: f64,
    eps: f64,
    swap: bool,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let a = borrow_tensor(anchor)?;
    let positive = borrow_tensor(positive)?;
    let negative = borrow_tensor(negative)?;
    let result = triplet_margin_loss_op(
        a.tensor(),
        positive.tensor(),
        negative.tensor(),
        margin,
        p,
        eps,
        swap,
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `log(1 + exp(-target * input))`, the smooth hinge, for a `target` of `+1` or `-1`.
#[pyfunction(name = "soft_margin_loss")]
#[pyo3(signature = (input, target, reduction=None))]
fn soft_margin_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let scores = borrow_tensor(input)?;
    let labels = borrow_tensor(target)?;
    let result = soft_margin_loss_op(
        scores.tensor(),
        labels.tensor(),
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// The negative log-likelihood of a Poisson observation. `log_input` says whether `input` is the log of the rate or the rate itself; `full` adds the Stirling term, which changes no gradient because it depends only on `target`.
#[pyfunction(name = "poisson_nll_loss")]
#[pyo3(signature = (input, target, log_input=true, full=false, eps=1e-8, reduction=None))]
fn poisson_nll_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    log_input: bool,
    full: bool,
    eps: f64,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let rate = borrow_tensor(input)?;
    let counts = borrow_tensor(target)?;
    let result = poisson_nll_loss_op(
        rate.tensor(),
        counts.tensor(),
        log_input,
        full,
        eps,
        reduction.unwrap_or("mean"),
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Connectionist temporal classification: the total probability of every alignment of `targets` to `log_probs`, for a model whose output is longer than its target and unaligned with it. `log_probs` is `(steps, batch, classes)` and is expected to be log probabilities already. `targets` is either a padded `(batch, length)` block or the rows concatenated into a vector, and may not contain the blank class. `reduction="mean"` divides each loss by its own target length before averaging. `zero_infinity` replaces the infinite loss of a target too long to fit its input, and its gradient, with zero.
#[pyfunction(name = "ctc_loss")]
#[pyo3(signature = (log_probs, targets, input_lengths, target_lengths, blank=0, reduction=None, zero_infinity=false))]
#[allow(clippy::too_many_arguments)]
fn ctc_loss_functional(
    log_probs: &Bound<PyAny>,
    targets: &Bound<PyAny>,
    input_lengths: &Bound<PyAny>,
    target_lengths: &Bound<PyAny>,
    blank: usize,
    reduction: Option<&str>,
    zero_infinity: bool,
) -> PyResult<PyTensor> {
    let probabilities = borrow_tensor(log_probs)?;
    let labels = PyTensor::from_python_value(targets)?;
    let inputs = PyTensor::from_python_value(input_lengths)?;
    let lengths = PyTensor::from_python_value(target_lengths)?;
    let result = ctc_loss_op(
        probabilities.tensor(),
        labels.tensor(),
        inputs.tensor(),
        lengths.tensor(),
        blank,
        reduction.unwrap_or("mean"),
        zero_infinity,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Read `input` at the normalised coordinates in `grid`, and differentiate with respect to both. `input` is `(batch, channels, height, width)` or `(batch, channels, depth, height, width)`; `grid` matches its rank and holds one coordinate per spatial axis in its last, in `x, y` (or `x, y, z`) order -- the reverse of the axes they index. Coordinates run from -1 to 1 across the input, with `align_corners` deciding whether those name the corner samples' centres or their outer edges. `padding_mode` says what lies outside: `"zeros"`, `"border"` or `"reflection"`. `mode="nearest"` has no gradient in the coordinates; `"bilinear"` is the one a spatial transformer can train through.
#[pyfunction(name = "grid_sample")]
#[pyo3(signature = (input, grid, mode="bilinear", padding_mode="zeros", align_corners=false))]
fn grid_sample_functional(
    input: &Bound<PyAny>,
    grid: &Bound<PyAny>,
    mode: &str,
    padding_mode: &str,
    align_corners: bool,
) -> PyResult<PyTensor> {
    let image = borrow_tensor(input)?;
    let coordinates = borrow_tensor(grid)?;
    let result = grid_sample_op(
        image.tensor(),
        coordinates.tensor(),
        SampleMode::from_name(mode).map_err(_convert_error)?,
        GridPadding::from_name(padding_mode).map_err(_convert_error)?,
        align_corners,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `log(cosh(prediction - target))`: smooth everywhere, and asymptotically linear like `l1_loss`.
#[pyfunction(name = "log_cosh_loss")]
#[pyo3(signature = (input, target, reduction=None))]
fn log_cosh_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let loss = LogCoshLoss::new(reduction);
    let result = loss
        .forward(prediction.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Binary cross-entropy, taking probabilities. Use the `_with_logits` form for raw scores.
#[pyfunction(name = "binary_cross_entropy")]
#[pyo3(signature = (input, target, reduction="mean"))]
fn binary_cross_entropy_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let loss = BCELoss::new(reduction);
    let result = loss
        .forward(prediction.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Binary cross-entropy taking raw scores, fusing the sigmoid so large-magnitude logits do not saturate.
#[pyfunction(name = "binary_cross_entropy_with_logits")]
#[pyo3(signature = (input, target, pos_weight=None, reduction="mean"))]
fn binary_cross_entropy_with_logits_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    pos_weight: Option<&Bound<PyAny>>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let logits = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let loss = match pos_weight {
        Some(w) => {
            let w = borrow_tensor(w)?;
            BCEWithLogitsLoss::with_pos_weight(reduction, w.tensor().clone())
        }
        None => BCEWithLogitsLoss::new(reduction),
    };
    let result = loss
        .forward(logits.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Base class for neural network modules
#[pyclass(name = "Module", subclass)]
pub struct PyModule {
    // This will be a trait object in practice
    // For now, we'll use an enum to handle different layer types
    inner: ModuleType,
}

/// Declare the module variants once and derive the uniform dispatch from that
/// single list. Every wrapped layer implements `Layer` (and therefore `Module`
/// through the blanket impl), so forwarding, parameter queries, train/eval and
/// serialization are all just trait calls on the erased inner layer — only
/// genuinely variant-specific behavior (`__repr__`, the typed constructors and
/// downcast getters) is written out per variant.
macro_rules! module_types {
    ($($variant:ident($ty:ty)),+ $(,)?) => {
        /// Payloads are boxed uniformly: the layer structs differ in size by
        /// almost 2x (`MultiheadAttention` carries eight tensors), and an
        /// unboxed enum would pad every module — including a bare `ReLU` — out
        /// to the largest one. Modules are constructed once, so the single
        /// allocation is free in exchange.
        enum ModuleType {
            $($variant(Box<$ty>),)+
        }

        impl ModuleType {
            fn as_layer(&self) -> &dyn Layer {
                match self {
                    $(ModuleType::$variant(layer) => &**layer,)+
                }
            }

            fn as_module(&self) -> &dyn engine::nn::Module {
                match self {
                    $(ModuleType::$variant(layer) => &**layer,)+
                }
            }

            fn as_module_mut(&mut self) -> &mut dyn engine::nn::Module {
                match self {
                    $(ModuleType::$variant(layer) => &mut **layer,)+
                }
            }

        }
    };
}

module_types! {
    DenseLayer(DenseLayer),
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    Softmax(Softmax),
    LeakyReLU(LeakyReLU),
    Elu(ELU),
    Gelu(GELU),
    Sequential(Sequential),
    Conv2d(Conv2d),
    BatchNorm1d(BatchNorm1d),
    BatchNorm2d(BatchNorm2d),
    Dropout(Dropout),
    Dropout2d(Dropout2d),
    Embedding(Embedding),
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
    MultiheadAttention(MultiheadAttention),
    MaxPool2d(MaxPool2d),
    AvgPool2d(AvgPool2d),
    Recurrent(Recurrent),
    Conv1d(Conv1d),
    MaxPool1d(MaxPool1d),
    AvgPool1d(AvgPool1d),
    ConvTranspose2d(ConvTranspose2d),
    ConvTranspose1d(ConvTranspose1d),
    AdaptiveAvgPool2d(AdaptiveAvgPool2d),
    AdaptiveMaxPool2d(AdaptiveMaxPool2d),
    AdaptiveAvgPool1d(AdaptiveAvgPool1d),
    AdaptiveMaxPool1d(AdaptiveMaxPool1d),
    Upsample(Upsample),
}

#[pymethods]
impl PyModule {
    /// Forward pass through the module
    fn forward(&mut self, input: &Bound<PyAny>) -> PyResult<PyTensor> {
        let input_tensor = borrow_tensor(input)?;
        let result = self
            .inner
            .as_module_mut()
            .forward(input_tensor.tensor())
            .map_err(_convert_error)?;

        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&mut self, input: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(input)
    }

    /// Get all parameters of the module
    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .as_layer()
            .parameters()
            .into_iter()
            .map(|tensor| PyTensor::from_tensor(tensor.clone()))
            .collect()
    }

    /// Set module to training mode
    fn train(&mut self) {
        self.inner.as_module_mut().train()
    }

    /// Set module to evaluation mode
    fn eval(&mut self) {
        self.inner.as_module_mut().eval()
    }

    /// Get number of parameters
    fn num_parameters(&self) -> usize {
        self.inner.as_layer().num_parameters()
    }

    /// Get detailed parameter statistics
    fn parameter_stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let layer: &dyn Layer = self.inner.as_layer();
        let stats = LayerUtils::parameter_stats(layer);
        let dict = PyDict::new(py);
        dict.set_item("total_parameters", stats.total_parameters)?;
        dict.set_item("trainable_parameters", stats.trainable_parameters)?;
        dict.set_item("non_trainable_parameters", stats.non_trainable_parameters)?;
        dict.set_item("parameter_count_by_tensor", stats.parameter_count_by_tensor)?;
        Ok(dict.into())
    }

    /// Get memory usage information
    fn memory_usage(&self, py: Python) -> PyResult<Py<PyAny>> {
        let layer: &dyn Layer = self.inner.as_layer();
        let usage = LayerUtils::memory_usage(layer);
        let dict = PyDict::new(py);
        dict.set_item("total_bytes", usage.total_bytes)?;
        let dtype_dict = PyDict::new(py);
        for (dtype, bytes) in usage.bytes_by_dtype {
            dtype_dict.set_item(crate::dtype::dtype_to_python_string(dtype), bytes)?;
        }
        dict.set_item("bytes_by_dtype", dtype_dict)?;
        Ok(dict.into())
    }

    /// Generate summary
    #[pyo3(signature = (name=None))]
    fn summary(&self, name: Option<&str>) -> PyResult<String> {
        match &self.inner {
            ModuleType::Sequential(model) => Ok(SequentialUtils::model_summary(model, name)),
            _ => {
                let layer: &dyn Layer = self.inner.as_layer();
                let owned;
                let layer_name = match name {
                    Some(n) => n,
                    None => {
                        owned = self.__repr__();
                        &owned
                    }
                };
                Ok(LayerUtils::layer_summary(layer, layer_name))
            }
        }
    }

    /// Rough forward-pass memory sketch for Sequential models.
    ///
    /// `parameter_memory` is exact. The activation numbers are not: layers
    /// cannot report an output shape, so this charges every layer an
    /// activation the size of the input. A model that widens is under-counted
    /// and one that narrows is over-counted. Element width follows the model's
    /// own dtype. Returns a dict of byte counts; use it for order-of-magnitude
    /// budgeting, not to decide whether a model fits.
    fn forward_memory_estimate(
        &self,
        input_shape: Vec<usize>,
        batch_size: usize,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        if let ModuleType::Sequential(model) = &self.inner {
            let est = SequentialUtils::estimate_forward_memory(model, &input_shape, batch_size);
            let dict = PyDict::new(py);
            dict.set_item("parameter_memory", est.parameter_memory)?;
            dict.set_item(
                "estimated_activation_memory",
                est.estimated_activation_memory,
            )?;
            dict.set_item("estimated_total_memory", est.estimated_total_memory)?;
            dict.set_item("input_memory", est.input_memory)?;
            Ok(dict.into())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "forward_memory_estimate only valid for Sequential modules",
            ))
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        // Python spells its booleans `True`/`False`; Rust's `Display` gives
        // `true`/`false`, which is not valid Python in a `__repr__`.
        fn py_bool(value: bool) -> &'static str {
            if value { "True" } else { "False" }
        }

        match &self.inner {
            ModuleType::DenseLayer(layer) => format!(
                "DenseLayer(in_features={}, out_features={})",
                layer.in_features(),
                layer.out_features()
            ),
            ModuleType::ReLU(_) => "ReLU()".to_string(),
            ModuleType::Sigmoid(_) => "Sigmoid()".to_string(),
            ModuleType::Tanh(_) => "Tanh()".to_string(),
            ModuleType::Softmax(layer) => format!("Softmax(dim={:?})", layer.dim()),
            ModuleType::LeakyReLU(layer) => {
                format!("LeakyReLU(negative_slope={})", layer.negative_slope())
            }
            ModuleType::Elu(layer) => format!("ELU(alpha={})", layer.alpha()),
            ModuleType::Gelu(layer) => {
                if layer.is_approximate() {
                    "GELU()".to_string()
                } else {
                    "GELU(approximate=\"none\")".to_string()
                }
            }
            ModuleType::Sequential(_) => "Sequential(...)".to_string(),
            ModuleType::Conv2d(layer) => format!(
                "Conv2d(in_channels={}, out_channels={}, kernel_size={:?})",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ),
            ModuleType::ConvTranspose2d(layer) => format!(
                "ConvTranspose2d(in_channels={}, out_channels={}, kernel_size={:?})",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ),
            ModuleType::ConvTranspose1d(layer) => format!(
                "ConvTranspose1d(in_channels={}, out_channels={}, kernel_size={})",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ),
            ModuleType::Upsample(layer) => match (layer.size(), layer.scale_factor()) {
                (Some(size), _) => format!("Upsample(size={size:?}, mode={:?})", layer.mode()),
                (None, Some(factor)) => {
                    format!("Upsample(scale_factor={factor:?}, mode={:?})", layer.mode())
                }
                (None, None) => "Upsample(mode=?)".to_string(),
            },
            ModuleType::AdaptiveAvgPool2d(layer) => {
                format!("AdaptiveAvgPool2d(output_size={:?})", layer.output_size())
            }
            ModuleType::AdaptiveMaxPool2d(layer) => {
                format!("AdaptiveMaxPool2d(output_size={:?})", layer.output_size())
            }
            ModuleType::AdaptiveAvgPool1d(layer) => {
                format!("AdaptiveAvgPool1d(output_size={})", layer.output_size())
            }
            ModuleType::AdaptiveMaxPool1d(layer) => {
                format!("AdaptiveMaxPool1d(output_size={})", layer.output_size())
            }
            ModuleType::BatchNorm1d(layer) => {
                format!("BatchNorm1d(num_features={})", layer.num_features())
            }
            ModuleType::BatchNorm2d(layer) => {
                format!("BatchNorm2d(num_features={})", layer.num_features())
            }
            ModuleType::Dropout(layer) => format!("Dropout(p={})", layer.p()),
            ModuleType::Dropout2d(layer) => format!("Dropout2d(p={})", layer.p()),
            ModuleType::Embedding(layer) => format!(
                "Embedding(num_embeddings={}, embedding_dim={}, padding_idx={:?})",
                layer.num_embeddings(),
                layer.embedding_dim(),
                layer.padding_idx()
            ),
            ModuleType::LayerNorm(layer) => format!(
                "LayerNorm(normalized_shape={:?}, eps={}, elementwise_affine={})",
                layer.normalized_shape(),
                layer.eps(),
                py_bool(layer.elementwise_affine())
            ),
            ModuleType::RMSNorm(layer) => format!(
                "RMSNorm(normalized_shape={:?}, eps={}, elementwise_affine={})",
                layer.normalized_shape(),
                layer.eps(),
                py_bool(layer.elementwise_affine())
            ),
            ModuleType::MultiheadAttention(layer) => format!(
                "MultiheadAttention(embed_dim={}, num_heads={}, head_dim={}, is_causal={})",
                layer.embed_dim(),
                layer.num_heads(),
                layer.head_dim(),
                py_bool(layer.is_causal())
            ),
            ModuleType::MaxPool2d(layer) => format!(
                "MaxPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
                layer.kernel_size(),
                layer.stride(),
                layer.padding()
            ),
            ModuleType::AvgPool2d(layer) => format!(
                "AvgPool2d(kernel_size={:?}, stride={:?}, padding={:?}, count_include_pad={})",
                layer.kernel_size(),
                layer.stride(),
                layer.padding(),
                py_bool(layer.count_include_pad())
            ),
            ModuleType::Conv1d(layer) => format!(
                "Conv1d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={})",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size(),
                layer.stride(),
                layer.padding()
            ),
            ModuleType::MaxPool1d(layer) => format!(
                "MaxPool1d(kernel_size={}, stride={}, padding={})",
                layer.kernel_size(),
                layer.stride(),
                layer.padding()
            ),
            ModuleType::AvgPool1d(layer) => format!(
                "AvgPool1d(kernel_size={}, stride={}, padding={}, count_include_pad={})",
                layer.kernel_size(),
                layer.stride(),
                layer.padding(),
                py_bool(layer.count_include_pad())
            ),
            ModuleType::Recurrent(layer) => format!(
                "{}(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
                match layer.kind() {
                    CellKind::Lstm => "LSTM",
                    CellKind::Gru => "GRU",
                },
                layer.input_size(),
                layer.hidden_size(),
                layer.num_layers(),
                py_bool(layer.has_bias()),
                py_bool(layer.batch_first()),
                py_bool(layer.bidirectional())
            ),
        }
    }

    /// Save module state to a file (basic implementation)
    #[pyo3(signature = (path, format=None))]
    fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        // Build a SerializedModel with metadata and engine state_dict
        let state = self.inner.as_module().state_dict();

        let metadata = ModelMetadata::new("module".to_string(), "Module".to_string());
        let model = SerializedModel::new(metadata, state);
        match format.map(|s| s.to_lowercase()) {
            Some(ref s) if s == "json" => {
                ModelSerializer::save(&model, path, SerializationFormat::Json)
            }
            Some(ref s) if s == "bin" || s == "binary" => {
                ModelSerializer::save(&model, path, SerializationFormat::Binary)
            }
            Some(ref s) if s == "msgpack" || s == "messagepack" => {
                ModelSerializer::save(&model, path, SerializationFormat::MessagePack)
            }
            _ => ModelSerializer::save_auto(&model, path),
        }
        .map_err(_convert_error)
    }

    /// Load module state from a file (basic implementation)
    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    fn load_state_from(path: &str, format: Option<&str>) -> PyResult<PyStateDict> {
        let model = match format.map(|s| s.to_lowercase()) {
            Some(ref s) if s == "json" => ModelSerializer::load(path, SerializationFormat::Json),
            Some(ref s) if s == "bin" || s == "binary" => {
                ModelSerializer::load(path, SerializationFormat::Binary)
            }
            Some(ref s) if s == "msgpack" || s == "messagepack" => {
                ModelSerializer::load(path, SerializationFormat::MessagePack)
            }
            _ => ModelSerializer::load_auto(path),
        }
        .map_err(_convert_error)?;
        Ok(crate::serialization::PyStateDict::from_engine(
            model.state_dict,
        ))
    }

    /// Return a StateDict snapshot of this module
    fn state_dict(&self) -> PyStateDict {
        let state = self.inner.as_module().state_dict();
        crate::serialization::PyStateDict::from_engine(state)
    }

    /// Load a provided StateDict into this module
    #[pyo3(signature = (state, device=None))]
    fn load_state_dict(&mut self, state: &PyStateDict, device: Option<&PyDevice>) -> PyResult<()> {
        let dev = device
            .map(|d| crate::device::ensure_available(d.device()))
            .transpose()?;
        let sd_ref = crate::serialization::PyStateDict::inner_ref(state);
        self.inner
            .as_module_mut()
            .load_state_dict(sd_ref, dev)
            .map_err(_convert_error)
    }
}

impl PyModule {
    pub fn from_dense_layer(dense_layer: DenseLayer) -> Self {
        Self {
            inner: ModuleType::DenseLayer(Box::new(dense_layer)),
        }
    }

    pub fn from_relu(relu: ReLU) -> Self {
        Self {
            inner: ModuleType::ReLU(Box::new(relu)),
        }
    }

    pub fn from_sigmoid(sigmoid: Sigmoid) -> Self {
        Self {
            inner: ModuleType::Sigmoid(Box::new(sigmoid)),
        }
    }

    pub fn from_tanh(tanh: Tanh) -> Self {
        Self {
            inner: ModuleType::Tanh(Box::new(tanh)),
        }
    }

    pub fn from_softmax(softmax: Softmax) -> Self {
        Self {
            inner: ModuleType::Softmax(Box::new(softmax)),
        }
    }

    pub fn from_leaky_relu(leaky_relu: LeakyReLU) -> Self {
        Self {
            inner: ModuleType::LeakyReLU(Box::new(leaky_relu)),
        }
    }

    pub fn from_elu(elu: ELU) -> Self {
        Self {
            inner: ModuleType::Elu(Box::new(elu)),
        }
    }

    pub fn from_gelu(gelu: GELU) -> Self {
        Self {
            inner: ModuleType::Gelu(Box::new(gelu)),
        }
    }

    pub fn from_sequential(sequential: Sequential) -> Self {
        Self {
            inner: ModuleType::Sequential(Box::new(sequential)),
        }
    }

    pub fn from_conv2d(conv2d: Conv2d) -> Self {
        Self {
            inner: ModuleType::Conv2d(Box::new(conv2d)),
        }
    }

    pub fn from_upsample(layer: Upsample) -> Self {
        Self {
            inner: ModuleType::Upsample(Box::new(layer)),
        }
    }

    pub fn from_adaptive_avg_pool2d(layer: AdaptiveAvgPool2d) -> Self {
        Self {
            inner: ModuleType::AdaptiveAvgPool2d(Box::new(layer)),
        }
    }

    pub fn from_adaptive_max_pool2d(layer: AdaptiveMaxPool2d) -> Self {
        Self {
            inner: ModuleType::AdaptiveMaxPool2d(Box::new(layer)),
        }
    }

    pub fn from_adaptive_avg_pool1d(layer: AdaptiveAvgPool1d) -> Self {
        Self {
            inner: ModuleType::AdaptiveAvgPool1d(Box::new(layer)),
        }
    }

    pub fn from_adaptive_max_pool1d(layer: AdaptiveMaxPool1d) -> Self {
        Self {
            inner: ModuleType::AdaptiveMaxPool1d(Box::new(layer)),
        }
    }

    pub fn from_conv_transpose2d(layer: ConvTranspose2d) -> Self {
        Self {
            inner: ModuleType::ConvTranspose2d(Box::new(layer)),
        }
    }

    pub fn from_conv_transpose1d(layer: ConvTranspose1d) -> Self {
        Self {
            inner: ModuleType::ConvTranspose1d(Box::new(layer)),
        }
    }

    pub fn from_max_pool2d(max_pool2d: MaxPool2d) -> Self {
        Self {
            inner: ModuleType::MaxPool2d(Box::new(max_pool2d)),
        }
    }

    pub fn from_avg_pool2d(avg_pool2d: AvgPool2d) -> Self {
        Self {
            inner: ModuleType::AvgPool2d(Box::new(avg_pool2d)),
        }
    }

    pub fn from_batch_norm1d(batch_norm1d: BatchNorm1d) -> Self {
        Self {
            inner: ModuleType::BatchNorm1d(Box::new(batch_norm1d)),
        }
    }

    pub fn from_batch_norm2d(batch_norm2d: BatchNorm2d) -> Self {
        Self {
            inner: ModuleType::BatchNorm2d(Box::new(batch_norm2d)),
        }
    }

    pub fn from_dropout(dropout: Dropout) -> Self {
        Self {
            inner: ModuleType::Dropout(Box::new(dropout)),
        }
    }

    pub fn from_dropout2d(dropout: Dropout2d) -> Self {
        Self {
            inner: ModuleType::Dropout2d(Box::new(dropout)),
        }
    }

    pub fn from_embedding(embedding: Embedding) -> Self {
        Self {
            inner: ModuleType::Embedding(Box::new(embedding)),
        }
    }

    pub fn from_layer_norm(layer_norm: LayerNorm) -> Self {
        Self {
            inner: ModuleType::LayerNorm(Box::new(layer_norm)),
        }
    }

    pub fn from_rms_norm(rms_norm: RMSNorm) -> Self {
        Self {
            inner: ModuleType::RMSNorm(Box::new(rms_norm)),
        }
    }

    pub fn from_conv1d(conv1d: Conv1d) -> Self {
        Self {
            inner: ModuleType::Conv1d(Box::new(conv1d)),
        }
    }

    pub fn from_max_pool1d(layer: MaxPool1d) -> Self {
        Self {
            inner: ModuleType::MaxPool1d(Box::new(layer)),
        }
    }

    pub fn from_avg_pool1d(layer: AvgPool1d) -> Self {
        Self {
            inner: ModuleType::AvgPool1d(Box::new(layer)),
        }
    }

    pub fn from_recurrent(recurrent: Recurrent) -> Self {
        Self {
            inner: ModuleType::Recurrent(Box::new(recurrent)),
        }
    }

    pub fn from_multihead_attention(mha: MultiheadAttention) -> Self {
        Self {
            inner: ModuleType::MultiheadAttention(Box::new(mha)),
        }
    }

    /// Clone the inner layer into a boxed trait object. Written out per variant rather than derived from `module_types!` because `Sequential` is not `Clone` and is rejected outright.
    pub fn to_layer(&self) -> PyResult<Box<dyn Layer>> {
        let layer: Box<dyn Layer> = match &self.inner {
            ModuleType::DenseLayer(layer) => layer.clone(),
            ModuleType::ReLU(layer) => layer.clone(),
            ModuleType::Sigmoid(layer) => layer.clone(),
            ModuleType::Tanh(layer) => layer.clone(),
            ModuleType::Softmax(layer) => layer.clone(),
            ModuleType::LeakyReLU(layer) => layer.clone(),
            ModuleType::Elu(layer) => layer.clone(),
            ModuleType::Gelu(layer) => layer.clone(),
            ModuleType::Sequential(_) => {
                return Err(PyTypeError::new_err(
                    "Nested Sequential modules are not supported",
                ));
            }
            ModuleType::Conv2d(layer) => layer.clone(),
            ModuleType::BatchNorm1d(layer) => layer.clone(),
            ModuleType::BatchNorm2d(layer) => layer.clone(),
            ModuleType::Dropout(layer) => layer.clone(),
            ModuleType::Dropout2d(layer) => layer.clone(),
            ModuleType::Embedding(layer) => layer.clone(),
            ModuleType::MultiheadAttention(layer) => layer.clone(),
            ModuleType::LayerNorm(layer) => layer.clone(),
            ModuleType::RMSNorm(layer) => layer.clone(),
            ModuleType::MaxPool2d(layer) => layer.clone(),
            ModuleType::AvgPool2d(layer) => layer.clone(),
            ModuleType::Recurrent(layer) => layer.clone(),
            ModuleType::Conv1d(layer) => layer.clone(),
            ModuleType::MaxPool1d(layer) => layer.clone(),
            ModuleType::AvgPool1d(layer) => layer.clone(),
            ModuleType::ConvTranspose2d(layer) => layer.clone(),
            ModuleType::ConvTranspose1d(layer) => layer.clone(),
            ModuleType::AdaptiveAvgPool2d(layer) => layer.clone(),
            ModuleType::AdaptiveMaxPool2d(layer) => layer.clone(),
            ModuleType::AdaptiveAvgPool1d(layer) => layer.clone(),
            ModuleType::AdaptiveMaxPool1d(layer) => layer.clone(),
            ModuleType::Upsample(layer) => layer.clone(),
        };

        Ok(layer)
    }
}

/// DenseLayer (fully connected) layer
#[pyclass(name = "DenseLayer", extends = PyModule)]
pub struct PyDenseLayer;

#[pymethods]
impl PyDenseLayer {
    /// Create a new dense layer
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=None, device=None, dtype=None))]
    fn new(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let bias = bias.unwrap_or(true);
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let dense_layer = DenseLayer::new(in_features, out_features, bias, device, dtype)
            .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_dense_layer(dense_layer)).add_subclass(Self))
    }

    /// Get input features count
    #[getter]
    fn in_features(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::DenseLayer(layer) = &module.inner {
            Ok(layer.in_features())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Get output features count
    #[getter]
    fn out_features(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::DenseLayer(layer) = &module.inner {
            Ok(layer.out_features())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Get weight tensor
    #[getter]
    fn weight(slf: PyRef<Self>) -> PyResult<PyTensor> {
        let module = slf.as_ref();
        if let ModuleType::DenseLayer(layer) = &module.inner {
            Ok(PyTensor::from_tensor(layer.weight().clone()))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Get bias tensor
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<Option<PyTensor>> {
        let module = slf.as_ref();
        if let ModuleType::DenseLayer(layer) = &module.inner {
            Ok(layer.bias().map(|b| PyTensor::from_tensor(b.clone())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// ReLU activation layer
#[pyclass(name = "ReLU", extends = PyModule)]
pub struct PyReLU;
