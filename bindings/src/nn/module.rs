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
use std::sync::Arc;

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

/// A convolution's `padding` given by name rather than as zeros per side.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NamedPadding {
    /// No padding: only the positions where the whole kernel fits.
    Valid,
    /// Enough padding that a stride-1 output keeps the input's size.
    Same,
}

/// Read `padding` as one of the named modes, or `None` for a number or a
/// sequence of them, which the caller parses as it always has.
fn named_padding(padding: Option<&Bound<PyAny>>, op: &str) -> PyResult<Option<NamedPadding>> {
    let Some(name) = padding.and_then(|value| value.extract::<String>().ok()) else {
        return Ok(None);
    };
    match name.as_str() {
        "valid" => Ok(Some(NamedPadding::Valid)),
        "same" => Ok(Some(NamedPadding::Same)),
        other => Err(PyValueError::new_err(format!(
            "{op} padding must be 'valid', 'same' or a count of zeros per side, got '{other}'"
        ))),
    }
}

/// The input and symmetric padding a convolution runs with for `padding="same"`.
///
/// `same_padding` splits each axis's total so any odd zero goes after it; the
/// kernel pads symmetrically, so that extra zero is added to the input here.
/// `kernel` and `dilation` are per spatial axis, innermost last.
fn same_padded_input(
    op: &str,
    input: &engine::tensor::Tensor,
    kernel: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> PyResult<(engine::tensor::Tensor, Vec<usize>)> {
    if stride.iter().any(|&s| s != 1) {
        return Err(PyValueError::new_err(format!(
            "{op} padding='same' needs a stride of 1, got {stride:?}"
        )));
    }
    let splits: Vec<(usize, usize)> = kernel
        .iter()
        .zip(dilation)
        .map(|(&k, &d)| engine::ops::same_padding(k, d))
        .collect();
    let symmetric: Vec<usize> = splits.iter().map(|&(before, _)| before).collect();
    if splits.iter().all(|&(before, after)| before == after) {
        return Ok((input.clone(), symmetric));
    }
    // Innermost axis first, as `pad` takes it: nothing before, the odd zero after.
    let extra: Vec<usize> = splits
        .iter()
        .rev()
        .flat_map(|&(before, after)| [0, after - before])
        .collect();
    let padded = engine::ops::shape_ops::pad(
        input,
        &extra,
        engine::ops::shape_ops::PadMode::Constant,
        0.0,
    )
    .map_err(_convert_error)?;
    Ok((padded, symmetric))
}

/// 2-D cross-correlation of `input` with `weight`. `dilation` spaces the kernel taps apart; `groups` splits the channels into that many independent convolutions, so `groups=in_channels` is a depthwise convolution.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, dilation=None, groups=crate::Size(1)), text_signature = "(input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1)")]
fn conv2d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    dilation: Option<&Bound<PyAny>>,
    groups: crate::Size,
) -> PyResult<PyTensor> {
    let groups = groups.get();
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let stride = parse_pair_arg("stride", stride, (1, 1))?;
    let dilation = parse_pair_arg("dilation", dilation, (1, 1))?;
    let weight_dims = weight_tensor.tensor().shape().dims().to_vec();
    let (input_value, padding) = match named_padding(padding, "conv2d")? {
        Some(NamedPadding::Valid) => (input_tensor.tensor().clone(), (0, 0)),
        // A weight of the wrong rank is left for the kernel to report.
        Some(NamedPadding::Same) if weight_dims.len() == 4 => {
            let (padded, symmetric) = same_padded_input(
                "conv2d",
                input_tensor.tensor(),
                &weight_dims[2..],
                &[stride.0, stride.1],
                &[dilation.0, dilation.1],
            )?;
            (padded, (symmetric[0], symmetric[1]))
        }
        Some(NamedPadding::Same) => (input_tensor.tensor().clone(), (0, 0)),
        None => (
            input_tensor.tensor().clone(),
            parse_pair_arg("padding", padding, (0, 0))?,
        ),
    };
    let result = conv2d_op(
        &input_value,
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
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, output_padding=None, dilation=None, groups=crate::Size(1)), text_signature = "(input, weight, bias=None, stride=None, padding=None, output_padding=None, dilation=None, groups=1)")]
#[allow(clippy::too_many_arguments)]
fn conv_transpose2d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
    output_padding: Option<&Bound<PyAny>>,
    dilation: Option<&Bound<PyAny>>,
    groups: crate::Size,
) -> PyResult<PyTensor> {
    let groups = groups.get();
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
#[pyo3(signature = (input, weight, bias=None, stride=crate::Size(1), padding=crate::Size(0), output_padding=crate::Size(0), dilation=crate::Size(1), groups=crate::Size(1)), text_signature = "(input, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1, groups=1)")]
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: crate::Size,
    padding: crate::Size,
    output_padding: crate::Size,
    dilation: crate::Size,
    groups: crate::Size,
) -> PyResult<PyTensor> {
    let stride = stride.get();
    let padding = padding.get();
    let output_padding = output_padding.get();
    let dilation = dilation.get();
    let groups = groups.get();
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

/// 1-D cross-correlation of `input` with `weight`. See `conv2d` for `dilation`, `groups` and the named paddings.
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=crate::Size(1), padding=None, dilation=crate::Size(1), groups=crate::Size(1)), text_signature = "(input, weight, bias=None, stride=1, padding=None, dilation=1, groups=1)")]
fn conv1d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: crate::Size,
    padding: Option<&Bound<PyAny>>,
    dilation: crate::Size,
    groups: crate::Size,
) -> PyResult<PyTensor> {
    let stride = stride.get();
    let dilation = dilation.get();
    let groups = groups.get();
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let weight_dims = weight_tensor.tensor().shape().dims().to_vec();
    let (input_value, padding) = match named_padding(padding, "conv1d")? {
        Some(NamedPadding::Valid) => (input_tensor.tensor().clone(), 0),
        Some(NamedPadding::Same) if weight_dims.len() == 3 => {
            let (padded, symmetric) = same_padded_input(
                "conv1d",
                input_tensor.tensor(),
                &weight_dims[2..],
                &[stride],
                &[dilation],
            )?;
            (padded, symmetric[0])
        }
        Some(NamedPadding::Same) => (input_tensor.tensor().clone(), 0),
        None => {
            let zeros = match padding {
                None => 0,
                Some(value) => value.extract::<isize>()?,
            };
            if zeros < 0 {
                return Err(PyValueError::new_err("padding must be non-negative"));
            }
            (input_tensor.tensor().clone(), zeros as usize)
        }
    };
    let result = conv1d_op(
        &input_value,
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
#[pyo3(signature = (input, kernel_size, stride=None, padding=crate::Size(0), return_indices=false), text_signature = "(input, kernel_size, stride=None, padding=0, return_indices=False)")]
fn max_pool1d(
    py: Python<'_>,
    input: &Bound<PyAny>,
    kernel_size: crate::Size,
    stride: Option<crate::Size>,
    padding: crate::Size,
    return_indices: bool,
) -> PyResult<Py<PyAny>> {
    let kernel_size = kernel_size.get();
    let stride = stride.map(crate::Size::get);
    let padding = padding.get();
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
#[pyo3(signature = (input, kernel_size, stride=None, padding=crate::Size(0), count_include_pad=true), text_signature = "(input, kernel_size, stride=None, padding=0, count_include_pad=True)")]
fn avg_pool1d(
    input: &Bound<PyAny>,
    kernel_size: crate::Size,
    stride: Option<crate::Size>,
    padding: crate::Size,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    let kernel_size = kernel_size.get();
    let stride = stride.map(crate::Size::get);
    let padding = padding.get();
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
fn adaptive_avg_pool1d(input: &Bound<PyAny>, output_size: crate::Size) -> PyResult<PyTensor> {
    let output_size = output_size.get();
    let input_tensor = borrow_tensor(input)?;
    let result =
        adaptive_avg_pool1d_op(input_tensor.tensor(), output_size).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// 1-D adaptive max pooling over `[N, C, L]`.
#[pyfunction]
#[pyo3(signature = (input, output_size))]
fn adaptive_max_pool1d(input: &Bound<PyAny>, output_size: crate::Size) -> PyResult<PyTensor> {
    let output_size = output_size.get();
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
#[pyo3(signature = (input, target, reduction="mean"))]
fn mse_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let loss = MSELoss::new(reduction);
    let result = loss
        .forward(prediction.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `beta` was previously fixed at 1.0 -- `SmoothL1Loss` had no field for it. The default is unchanged.
#[pyfunction(name = "smooth_l1_loss")]
#[pyo3(signature = (input, target, reduction="mean", beta=1.0))]
fn smooth_l1_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
    beta: f64,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    // The op validates beta; huber and smooth-l1 differ by a factor of beta and
    // coincide only at 1.0, so this must not just forward to `huber_loss_op`.
    let result = smooth_l1_loss_op(prediction.tensor(), target_tensor.tensor(), beta, reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Squared error within `delta` of the target and linear beyond it, so outliers pull less than under `mse_loss`.
#[pyfunction(name = "huber_loss")]
#[pyo3(signature = (input, target, reduction="mean", delta=1.0))]
fn huber_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
    delta: f64,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
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
#[pyo3(signature = (input, target, reduction="mean"))]
fn l1_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let result = mae_loss_op(prediction.tensor(), target_tensor.tensor(), reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Kullback-Leibler divergence from `target` to `input`, both given as probabilities -- *not* as log-probabilities. A zero in `target` contributes nothing, as the definition requires.
#[pyfunction(name = "kl_div")]
#[pyo3(signature = (input, target, reduction="mean"))]
fn kl_div_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let result = kl_div_loss_op(prediction.tensor(), target_tensor.tensor(), reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Cross-entropy down-weighted on well-classified examples by `(1 - p) ** gamma`, for imbalanced classes.
#[pyfunction(name = "focal_loss")]
#[pyo3(signature = (input, target, alpha=0.25, gamma=2.0, reduction="mean"))]
fn focal_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    alpha: f64,
    gamma: f64,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
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
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction="mean"))]
fn margin_ranking_loss_functional(
    input1: &Bound<PyAny>,
    input2: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: &str,
) -> PyResult<PyTensor> {
    let left = borrow_tensor(input1)?;
    let right = borrow_tensor(input2)?;
    let labels = borrow_tensor(target)?;
    let result = margin_ranking_loss_op(
        left.tensor(),
        right.tensor(),
        labels.tensor(),
        margin,
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// The distance itself where `target` is `+1`, and `max(0, margin - distance)` where it is `-1`.
#[pyfunction(name = "hinge_embedding_loss")]
#[pyo3(signature = (input, target, margin=1.0, reduction="mean"))]
fn hinge_embedding_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: &str,
) -> PyResult<PyTensor> {
    let distances = borrow_tensor(input)?;
    let labels = borrow_tensor(target)?;
    let result = hinge_embedding_loss_op(distances.tensor(), labels.tensor(), margin, reduction)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `1 - cos(x1, x2)` where `target` is `+1`, and `max(0, cos(x1, x2) - margin)` where it is `-1`.
#[pyfunction(name = "cosine_embedding_loss")]
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction="mean"))]
fn cosine_embedding_loss_functional(
    input1: &Bound<PyAny>,
    input2: &Bound<PyAny>,
    target: &Bound<PyAny>,
    margin: f64,
    reduction: &str,
) -> PyResult<PyTensor> {
    let left = borrow_tensor(input1)?;
    let right = borrow_tensor(input2)?;
    let labels = borrow_tensor(target)?;
    let result = cosine_embedding_loss_op(
        left.tensor(),
        right.tensor(),
        labels.tensor(),
        margin,
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `max(0, d(anchor, positive) - d(anchor, negative) + margin)`. With `swap`, the negative distance is the smaller of `d(anchor, negative)` and `d(positive, negative)`, so a triplet whose positive sits closest to the negative still counts as a violation.
#[pyfunction(name = "triplet_margin_loss")]
#[pyo3(signature = (anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=false, reduction="mean"))]
#[allow(clippy::too_many_arguments)]
fn triplet_margin_loss_functional(
    anchor: &Bound<PyAny>,
    positive: &Bound<PyAny>,
    negative: &Bound<PyAny>,
    margin: f64,
    p: f64,
    eps: f64,
    swap: bool,
    reduction: &str,
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
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// `log(1 + exp(-target * input))`, the smooth hinge, for a `target` of `+1` or `-1`.
#[pyfunction(name = "soft_margin_loss")]
#[pyo3(signature = (input, target, reduction="mean"))]
fn soft_margin_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let scores = borrow_tensor(input)?;
    let labels = borrow_tensor(target)?;
    let result =
        soft_margin_loss_op(scores.tensor(), labels.tensor(), reduction).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// The negative log-likelihood of a Poisson observation. `log_input` says whether `input` is the log of the rate or the rate itself; `full` adds the Stirling term, which changes no gradient because it depends only on `target`.
#[pyfunction(name = "poisson_nll_loss")]
#[pyo3(signature = (input, target, log_input=true, full=false, eps=1e-8, reduction="mean"))]
fn poisson_nll_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    log_input: bool,
    full: bool,
    eps: f64,
    reduction: &str,
) -> PyResult<PyTensor> {
    let rate = borrow_tensor(input)?;
    let counts = borrow_tensor(target)?;
    let result = poisson_nll_loss_op(
        rate.tensor(),
        counts.tensor(),
        log_input,
        full,
        eps,
        reduction,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Connectionist temporal classification: the total probability of every alignment of `targets` to `log_probs`, for a model whose output is longer than its target and unaligned with it. `log_probs` is `(steps, batch, classes)` and is expected to be log probabilities already. `targets` is either a padded `(batch, length)` block or the rows concatenated into a vector, and may not contain the blank class. `reduction="mean"` divides each loss by its own target length before averaging. `zero_infinity` replaces the infinite loss of a target too long to fit its input, and its gradient, with zero.
#[pyfunction(name = "ctc_loss")]
#[pyo3(signature = (log_probs, targets, input_lengths, target_lengths, blank=crate::Size(0), reduction="mean", zero_infinity=false), text_signature = "(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)")]
#[allow(clippy::too_many_arguments)]
fn ctc_loss_functional(
    log_probs: &Bound<PyAny>,
    targets: &Bound<PyAny>,
    input_lengths: &Bound<PyAny>,
    target_lengths: &Bound<PyAny>,
    blank: crate::Size,
    reduction: &str,
    zero_infinity: bool,
) -> PyResult<PyTensor> {
    let blank = blank.get();
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
        reduction,
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
#[pyo3(signature = (input, target, reduction="mean"))]
fn log_cosh_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: &str,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
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
#[pyclass(name = "Module", module = "minitensor.nn", subclass)]
pub struct PyModule {
    inner: SharedModule,
}

/// A module's layer, held by its Python object and by the `Sequential` it
/// was added to, if any.
///
/// A `Sequential` used to hold a clone of each layer it was given. The clone
/// shared the parameters' storage but had flags and buffers of its own, so
/// the object the caller kept was half of the same layer:
/// `model = Sequential([backbone, head]); backbone.requires_grad_(False)`
/// left `model` training the backbone, `backbone.eval()` left its dropout
/// running inside `model`, and the backbone's running statistics stopped at
/// the values they had when it was added. Now both hold this, one layer.
///
/// Holding one layer from two places means `&mut` access that the borrow
/// checker cannot see, and [`ModuleCell`] is where that is made sound: see
/// its documentation for the argument.
struct SharedModule(Arc<ModuleCell>);

/// The layer and the bookkeeping that makes sharing it sound.
///
/// Every access happens with the GIL held, and with it held nothing else
/// touches a module: none of the methods that read or write a layer run
/// Python code, so they cannot be re-entered either. The exception is a
/// forward pass, which may release the GIL inside an operation, letting
/// another thread in while it still holds the layer mutably.
///
/// So a forward pass takes `tree`, the lock shared by every module in one
/// model -- a top-level module and everything added to it, at any depth --
/// and holds it throughout. Any other access first checks that it is not
/// held. A forward pass can only start with the GIL held (from Python) or
/// under a lock its model's forward already holds (a `Sequential` running
/// its children), so an access that found the lock free keeps the model
/// to itself until it returns.
///
/// The model is a tree: a module is added to at most one `Sequential`, and
/// never to one inside itself, so no two paths from a model reach one
/// layer and collecting `&mut` references across it cannot alias.
struct ModuleCell {
    layer: std::cell::UnsafeCell<ModuleType>,
    tree: std::sync::Mutex<Arc<std::sync::atomic::AtomicBool>>,
    /// The `Sequential` this was added to, while it exists.
    parent: std::sync::Mutex<std::sync::Weak<ModuleCell>>,
    /// What was added to this one, in order: each child's layer, for moving
    /// it onto the lock of whatever this one is added to, and the Python
    /// object it was added as, which is what indexing the `Sequential`
    /// returns.
    children: std::sync::Mutex<Vec<Child>>,
    /// Whether the module is in training mode. Only some layers behave
    /// differently in the two modes and keep a flag of their own, so the
    /// mode of any module is kept here, where every module has one.
    training: std::sync::atomic::AtomicBool,
}

struct Child {
    cell: Arc<ModuleCell>,
    object: Py<PyAny>,
}

// SAFETY: see `ModuleCell`: concurrent access is excluded by the GIL, and a
// forward pass that releases it holds the model's `tree` lock, which every
// other access checks first.
unsafe impl Sync for ModuleCell {}

impl ModuleCell {
    fn new(layer: ModuleType) -> Arc<Self> {
        Arc::new(Self {
            layer: std::cell::UnsafeCell::new(layer),
            tree: std::sync::Mutex::new(Arc::new(std::sync::atomic::AtomicBool::new(false))),
            parent: std::sync::Mutex::new(std::sync::Weak::new()),
            children: std::sync::Mutex::new(Vec::new()),
            training: std::sync::atomic::AtomicBool::new(true),
        })
    }

    fn tree(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.tree.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_training(&self, training: bool) {
        self.training
            .store(training, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn in_forward(&self) -> bool {
        self.tree().load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether `cell` is this one or anywhere inside it.
    fn contains(self: &Arc<Self>, cell: &Arc<ModuleCell>) -> bool {
        Arc::ptr_eq(self, cell)
            || self
                .children
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .any(|child| child.cell.contains(cell))
    }

    fn join_tree(&self, tree: &Arc<std::sync::atomic::AtomicBool>) {
        *self.tree.lock().unwrap_or_else(|e| e.into_inner()) = tree.clone();
        for child in self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            child.cell.join_tree(tree);
        }
    }

    /// # Safety
    ///
    /// The caller holds the GIL and has checked the model is not in a
    /// forward pass, or is the forward pass holding its lock.
    unsafe fn layer(&self) -> &ModuleType {
        unsafe { &*self.layer.get() }
    }

    /// # Safety
    ///
    /// As [`Self::layer`], and no other reference into this layer is live.
    #[allow(clippy::mut_from_ref)]
    unsafe fn layer_mut(&self) -> &mut ModuleType {
        unsafe { &mut *self.layer.get() }
    }
}

/// When a `Sequential` is dropped its children stand alone again: each takes
/// a lock of its own and can be added to another.
impl Drop for ModuleCell {
    fn drop(&mut self) {
        let children = self.children.get_mut().unwrap_or_else(|e| e.into_inner());
        for child in children.drain(..) {
            child
                .cell
                .join_tree(&Arc::new(std::sync::atomic::AtomicBool::new(false)));
        }
    }
}

impl SharedModule {
    fn new(layer: ModuleType) -> Self {
        Self(ModuleCell::new(layer))
    }

    fn check_idle(&self) -> PyResult<()> {
        if self.0.in_forward() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "this module is in a forward pass on another thread; use a \
                 model from one thread at a time",
            ));
        }
        Ok(())
    }

    /// The layer, for a Python method to read.
    fn get(&self) -> PyResult<&ModuleType> {
        self.check_idle()?;
        // SAFETY: GIL held (every caller is a Python method) and the model is
        // not in a forward pass.
        Ok(unsafe { self.0.layer() })
    }

    /// The layer, for a Python method to change.
    fn get_mut(&mut self) -> PyResult<&mut ModuleType> {
        self.check_idle()?;
        // SAFETY: as `get`, and `&mut self` excludes this object's other
        // methods; the other holder, a `Sequential`, only reaches the layer
        // from its own methods, which cannot be running (GIL held, not in a
        // forward pass).
        Ok(unsafe { self.0.layer_mut() })
    }

    /// Run `f` as this module's forward pass, holding its model's lock.
    fn forward<T>(&mut self, f: impl FnOnce(&mut ModuleType) -> T) -> PyResult<T> {
        let tree = self.0.tree();
        if tree
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "this module is already in a forward pass, on another thread or \
                 as part of the model it belongs to",
            ));
        }
        struct Release(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Release);
            }
        }
        let _release = Release(tree);
        // SAFETY: the model's lock is held, and `&mut self` excludes this
        // object's other methods.
        Ok(f(unsafe { self.0.layer_mut() }))
    }

    /// The layer, to be held by a `Sequential` as its child.
    ///
    /// Refused for a module that already belongs to one, and for one that
    /// `parent` is inside: either would give the model two paths to one
    /// layer. `object` is the Python object this is held by.
    fn adopt_into(&self, object: Py<PyAny>, parent: &SharedModule) -> PyResult<Box<dyn Layer>> {
        if self.0.in_forward() || parent.0.in_forward() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "a module cannot be added to a Sequential while either is in a \
                 forward pass",
            ));
        }
        if self.0.contains(&parent.0) {
            return Err(PyValueError::new_err(
                "a Sequential cannot hold itself, or a module it is inside",
            ));
        }
        let mut own_parent = self.0.parent.lock().unwrap_or_else(|e| e.into_inner());
        if own_parent.strong_count() > 0 {
            return Err(PyValueError::new_err(
                "this module already belongs to a Sequential, and a module can \
                 belong to only one; use copy.deepcopy(module) for a second, \
                 independent one",
            ));
        }
        *own_parent = Arc::downgrade(&parent.0);
        drop(own_parent);
        self.0.join_tree(&parent.0.tree());
        parent
            .0
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Child {
                cell: self.0.clone(),
                object,
            });
        Ok(Box::new(SharedChild(self.0.clone())))
    }
}

/// A module as a `Sequential`'s child: the same layer its Python object
/// holds.
///
/// Reached only from the `Sequential`'s own methods, so the access checks
/// `SharedModule` makes on the way in cover these too.
struct SharedChild(Arc<ModuleCell>);

impl SharedChild {
    fn layer(&self) -> &dyn Layer {
        // SAFETY: see the type's documentation.
        unsafe { self.0.layer() }.as_layer()
    }

    fn layer_mut(&mut self) -> &mut dyn Layer {
        // SAFETY: see the type's documentation; the model is a tree, so this
        // is the only path to the layer from the `Sequential` being walked.
        unsafe { self.0.layer_mut() }.as_layer_mut()
    }
}

impl Layer for SharedChild {
    fn forward(
        &mut self,
        input: &engine::tensor::Tensor,
    ) -> engine::error::Result<engine::tensor::Tensor> {
        // SAFETY: see the type's documentation.
        unsafe { self.0.layer() }.check_input_dtype(input)?;
        self.layer_mut().forward(input)
    }

    fn parameters(&self) -> Vec<&engine::tensor::Tensor> {
        self.layer().parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut engine::tensor::Tensor> {
        self.layer_mut().parameters_mut()
    }

    fn buffers(&self) -> Vec<&engine::tensor::Tensor> {
        self.layer().buffers()
    }

    fn buffers_mut(&mut self) -> Vec<&mut engine::tensor::Tensor> {
        self.layer_mut().buffers_mut()
    }

    /// A copy with a layer of its own, as every other layer's is: this is
    /// what copying a `Sequential` copies its children with.
    fn clone_layer(&self) -> Option<Box<dyn Layer>> {
        // SAFETY: see the type's documentation.
        let copy = unsafe { self.0.layer() }.try_clone()?;
        Some(Box::new(SharedChild(ModuleCell::new(copy))))
    }

    fn train(&mut self) {
        self.0.set_training(true);
        self.layer_mut().train()
    }

    fn eval(&mut self) {
        self.0.set_training(false);
        self.layer_mut().eval()
    }

    fn num_parameters(&self) -> usize {
        self.layer().num_parameters()
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, &engine::tensor::Tensor> {
        self.layer().named_parameters()
    }

    fn named_parameters_mut(
        &mut self,
    ) -> std::collections::HashMap<String, &mut engine::tensor::Tensor> {
        self.layer_mut().named_parameters_mut()
    }

    fn named_buffers(&self) -> std::collections::HashMap<String, &engine::tensor::Tensor> {
        self.layer().named_buffers()
    }

    fn named_buffers_mut(
        &mut self,
    ) -> std::collections::HashMap<String, &mut engine::tensor::Tensor> {
        self.layer_mut().named_buffers_mut()
    }
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
            fn variant_name(&self) -> &'static str {
                match self {
                    $(ModuleType::$variant(_) => stringify!($variant),)+
                }
            }

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

            fn as_layer_mut(&mut self) -> &mut dyn Layer {
                match self {
                    $(ModuleType::$variant(layer) => &mut **layer,)+
                }
            }

        }
    };
}

impl ModuleType {
    /// The Python class this layer is.
    fn class_name(&self) -> &'static str {
        match self {
            ModuleType::Elu(_) => "ELU",
            ModuleType::Gelu(_) => "GELU",
            ModuleType::Recurrent(layer) => match layer.kind() {
                CellKind::Lstm => "LSTM",
                CellKind::Gru => "GRU",
            },
            other => other.variant_name(),
        }
    }

    /// Refuse an input whose dtype differs from the layer's parameters.
    ///
    /// Checked here, at the layer, rather than left to the first operation
    /// inside it: that one reports its own operands, so a float64 layer given
    /// a float32 input read "expected Float32, got Float64" -- the input's dtype
    /// as the expectation, and no word of which layer. An `Embedding` reads
    /// integer indices, and a `Sequential` is checked child by child.
    fn check_input_dtype(&self, input: &engine::tensor::Tensor) -> engine::error::Result<()> {
        self.check_dtype_of(input, "input")
    }

    /// [`Self::check_input_dtype`] for any tensor the layer is given, `what`
    /// naming it in the message: a recurrent layer's initial states are
    /// checked the same way as its input.
    pub(crate) fn check_dtype_of(
        &self,
        input: &engine::tensor::Tensor,
        what: &str,
    ) -> engine::error::Result<()> {
        if matches!(self, ModuleType::Sequential(_) | ModuleType::Embedding(_)) {
            return Ok(());
        }
        let parameters = self.as_layer().parameters();
        let Some(parameter) = parameters.first() else {
            return Ok(());
        };
        if parameter.dtype() == input.dtype() {
            return Ok(());
        }
        let wanted = format!("{:?}", parameter.dtype()).to_lowercase();
        let given = format!("{:?}", input.dtype()).to_lowercase();
        Err(engine::error::MinitensorError::type_mismatch_with_context(
            wanted.clone(),
            given.clone(),
            format!(
                "{} has {wanted} parameters and was given a {given} {what}; to \
                 keep the {what} as it is, convert the model with .astype('{given}')",
                self.class_name()
            ),
        ))
    }
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
            .forward(|layer| {
                layer.check_input_dtype(input_tensor.tensor())?;
                layer.as_module_mut().forward(input_tensor.tensor())
            })?
            .map_err(_convert_error)?;

        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&mut self, input: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(input)
    }

    /// Handles to every parameter of the module.
    ///
    /// Each handle shares its parameter's storage and identity, so an update
    /// made through it -- an optimizer's step -- is the module's update. Its
    /// `requires_grad` flag is the handle's own, though: setting it changes
    /// what that handle does, not whether the module trains. Freeze or
    /// unfreeze the module itself with `requires_grad_`.
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(self
            .inner
            .get()?
            .as_layer()
            .parameters()
            .into_iter()
            .map(|tensor| PyTensor::from_tensor(tensor.clone()))
            .collect())
    }

    /// `(name, parameter)` for every parameter, in the order `parameters()`
    /// gives them and under the names `state_dict()` uses -- `0.weight` for
    /// the weight of a `Sequential`'s first layer -- so a part can be found,
    /// frozen or inspected by name. Each is a handle, as `parameters()`
    /// returns.
    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let layer = self.inner.get()?.as_layer();
        let names: std::collections::HashMap<_, _> = layer
            .named_parameters()
            .into_iter()
            .map(|(name, tensor)| (tensor.id(), name))
            .collect();
        Ok(layer
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| {
                let name = names
                    .get(&tensor.id())
                    .cloned()
                    .unwrap_or_else(|| format!("param_{index}"));
                (name, PyTensor::from_tensor(tensor.clone()))
            })
            .collect())
    }

    /// Handles to the module's buffers: state it keeps and saves but does not
    /// train, such as a batch norm's running statistics.
    fn buffers(&self) -> PyResult<Vec<PyTensor>> {
        Ok(self
            .inner
            .get()?
            .as_layer()
            .buffers()
            .into_iter()
            .map(|tensor| PyTensor::from_tensor(tensor.clone()))
            .collect())
    }

    /// `(name, buffer)` for every buffer, in the order `buffers()` gives them
    /// and under the names `state_dict()` uses.
    fn named_buffers(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let layer = self.inner.get()?.as_layer();
        let names: std::collections::HashMap<_, _> = layer
            .named_buffers()
            .into_iter()
            .map(|(name, tensor)| (tensor.id(), name))
            .collect();
        Ok(layer
            .buffers()
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| {
                let name = names
                    .get(&tensor.id())
                    .cloned()
                    .unwrap_or_else(|| format!("buffer_{index}"));
                (name, PyTensor::from_tensor(tensor.clone()))
            })
            .collect())
    }

    /// Convert every parameter and floating-point buffer to `dtype`, in place,
    /// and return the module.
    ///
    /// Only the float dtypes are accepted: parameters are trained through
    /// gradients, which an integer has none of. Each parameter keeps its
    /// `requires_grad`. The converted tensors are new ones, so an optimizer
    /// built over the module beforehand still holds the old ones -- build it
    /// after converting.
    fn astype<'py>(slf: Bound<'py, Self>, dtype: &str) -> PyResult<Bound<'py, Self>> {
        let target = dtype::parse_dtype(dtype)?;
        if !target.is_float() {
            return Err(PyValueError::new_err(format!(
                "a module's parameters must be floating point; astype takes \
                 'float32' or 'float64', got '{dtype}'"
            )));
        }
        {
            let mut this = slf.borrow_mut();
            let layer = this.inner.get_mut()?.as_layer_mut();
            for parameter in layer.parameters_mut() {
                if parameter.dtype() != target {
                    let trainable = parameter.requires_grad();
                    *parameter = parameter
                        .detach()
                        .astype(target)
                        .map_err(_convert_error)?
                        .requires_grad_(trainable);
                }
            }
            for buffer in layer.buffers_mut() {
                if buffer.dtype() != target && buffer.dtype().is_float() {
                    *buffer = buffer.detach().astype(target).map_err(_convert_error)?;
                }
            }
        }
        Ok(slf)
    }

    /// Clear the gradient of every trainable tensor this module owns.
    ///
    /// The reference has promised this since the module surface was written,
    /// and nothing implemented it: a reader following `layer.zero_grad()` got
    /// an `AttributeError`. It is `optimizer.zero_grad()` without an
    /// optimizer, for zeroing one branch of a model or for a loop that does
    /// its own stepping.
    #[pyo3(signature = (set_to_none=false))]
    fn zero_grad(&self, set_to_none: bool) -> PyResult<()> {
        for parameter in self.inner.get()?.as_layer().parameters() {
            let mut owned = parameter.clone();
            owned.zero_grad(set_to_none);
        }
        Ok(())
    }

    /// Set whether every parameter of this module takes a gradient, and
    /// return the module.
    ///
    /// `requires_grad_(False)` freezes it: a forward pass records nothing for
    /// its parameters, a backward pass leaves them without a gradient, and an
    /// optimizer holding them steps past them. `requires_grad_()` makes it
    /// trainable again. The parameters keep their identity and storage, so an
    /// optimizer built before the change still reaches them after it.
    ///
    /// This is the way to freeze: the handles `parameters()` returns carry
    /// flags of their own, and setting one does not reach the module. Buffers
    /// never take a gradient and are left alone.
    #[pyo3(signature = (requires_grad=true))]
    fn requires_grad_<'py>(
        slf: Bound<'py, Self>,
        requires_grad: bool,
    ) -> PyResult<Bound<'py, Self>> {
        for parameter in slf
            .borrow_mut()
            .inner
            .get_mut()?
            .as_layer_mut()
            .parameters_mut()
        {
            *parameter = parameter.clone().requires_grad_(requires_grad);
        }
        Ok(slf)
    }

    /// Set module to training mode
    ///
    /// `train(False)` is `eval()`. Returns the module, so a call can end a
    /// chain: `model = build().train()`.
    #[pyo3(signature = (mode=true))]
    fn train(slf: Bound<'_, Self>, mode: bool) -> PyResult<Bound<'_, Self>> {
        {
            let mut this = slf.borrow_mut();
            let layer = this.inner.get_mut()?.as_module_mut();
            if mode {
                layer.train()
            } else {
                layer.eval()
            }
            this.inner.0.set_training(mode);
        }
        Ok(slf)
    }

    /// Set module to evaluation mode
    ///
    /// Returns the module, as `train` does.
    fn eval(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        Self::train(slf, false)
    }

    /// Whether the module is in training mode: `True` from construction
    /// until `eval()`, and again after `train()`. Setting either on a
    /// `Sequential` sets it on everything inside, which reads it back here.
    #[getter]
    fn training(&self) -> bool {
        self.inner.0.is_training()
    }

    /// Get number of parameters
    fn num_parameters(&self) -> PyResult<usize> {
        Ok(self.inner.get()?.as_layer().num_parameters())
    }

    /// Get detailed parameter statistics
    fn parameter_stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let layer: &dyn Layer = self.inner.get()?.as_layer();
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
        let layer: &dyn Layer = self.inner.get()?.as_layer();
        let usage = LayerUtils::memory_usage(layer);
        let dict = PyDict::new(py);
        dict.set_item("total_bytes", usage.total_bytes)?;
        dict.set_item("parameter_bytes", usage.parameter_bytes)?;
        dict.set_item("buffer_bytes", usage.buffer_bytes)?;
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
        match self.inner.get()? {
            ModuleType::Sequential(model) => Ok(SequentialUtils::model_summary(model, name)),
            _ => {
                let layer: &dyn Layer = self.inner.get()?.as_layer();
                let owned;
                let layer_name = match name {
                    Some(n) => n,
                    None => {
                        owned = self.__repr__()?;
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
        input_shape: Vec<crate::Size>,
        batch_size: crate::Size,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let input_shape: Vec<usize> = input_shape.into_iter().map(crate::Size::get).collect();
        let batch_size = batch_size.get();
        if let ModuleType::Sequential(model) = self.inner.get()? {
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

    /// String representation: the constructor call that builds this layer.
    ///
    /// Required arguments always appear; optional ones only when they differ
    /// from their defaults, so two layers that behave differently never print
    /// the same. Values are written as Python spells them.
    fn __repr__(&self) -> PyResult<String> {
        /// Accumulates `, name=value` for the arguments that are not at their
        /// defaults.
        struct Args(String);
        impl Args {
            fn new(required: String) -> Self {
                Self(required)
            }
            fn unless<T: PartialEq>(
                mut self,
                name: &str,
                value: T,
                default: T,
                text: String,
            ) -> Self {
                if value != default {
                    self.0.push_str(&format!(", {name}={text}"));
                }
                self
            }
            fn call(self, class: &str) -> String {
                format!("{class}({})", self.0)
            }
        }
        // Python spells its booleans `True`/`False`; Rust's `Display` gives
        // `true`/`false`, which is not valid Python in a `__repr__`.
        fn py_bool(value: bool) -> String {
            if value { "True" } else { "False" }.to_string()
        }
        fn pair(value: (usize, usize)) -> String {
            format!("({}, {})", value.0, value.1)
        }
        /// A float as a Python literal: `1e-05` style is not needed, but a
        /// whole number must keep its point, and `{:?}` does both.
        fn float(value: f64) -> String {
            format!("{value:?}")
        }
        /// One value for every axis prints as that value; otherwise a tuple.
        fn per_axis<T: std::fmt::Debug + PartialEq>(values: &[T]) -> String {
            match values {
                [single] => format!("{single:?}"),
                _ => format!(
                    "({})",
                    values
                        .iter()
                        .map(|v| format!("{v:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
        fn has_bias(layer: &dyn Layer) -> bool {
            layer
                .named_parameters()
                .keys()
                .any(|name| name.contains("bias"))
        }
        fn mode_name(mode: engine::ops::interpolate::InterpolateMode) -> &'static str {
            match mode {
                engine::ops::interpolate::InterpolateMode::Nearest => "nearest",
                engine::ops::interpolate::InterpolateMode::Linear => "linear",
            }
        }

        let inner = self.inner.get()?;
        let bias = has_bias(inner.as_layer());
        Ok(match inner {
            ModuleType::DenseLayer(layer) => Args::new(format!(
                "in_features={}, out_features={}",
                layer.in_features(),
                layer.out_features()
            ))
            .unless("bias", bias, true, py_bool(bias))
            .call("DenseLayer"),
            ModuleType::ReLU(_) => "ReLU()".to_string(),
            ModuleType::Sigmoid(_) => "Sigmoid()".to_string(),
            ModuleType::Tanh(_) => "Tanh()".to_string(),
            ModuleType::Softmax(layer) => match layer.dim() {
                Some(dim) => format!("Softmax(dim={dim})"),
                None => "Softmax()".to_string(),
            },
            ModuleType::LeakyReLU(layer) => {
                format!(
                    "LeakyReLU(negative_slope={})",
                    float(layer.negative_slope())
                )
            }
            ModuleType::Elu(layer) => format!("ELU(alpha={})", float(layer.alpha())),
            ModuleType::Gelu(layer) => {
                if layer.is_approximate() {
                    "GELU()".to_string()
                } else {
                    "GELU(approximate=\"none\")".to_string()
                }
            }
            ModuleType::Sequential(_) => self.sequential_repr()?,
            ModuleType::Conv2d(layer) => Args::new(format!(
                "in_channels={}, out_channels={}, kernel_size={}",
                layer.in_channels(),
                layer.out_channels(),
                pair(layer.kernel_size())
            ))
            .unless("stride", layer.stride(), (1, 1), pair(layer.stride()))
            .unless(
                "padding",
                layer.is_same_padding(),
                false,
                "'same'".to_string(),
            )
            .unless(
                "padding",
                if layer.is_same_padding() {
                    (0, 0)
                } else {
                    layer.padding()
                },
                (0, 0),
                pair(layer.padding()),
            )
            .unless("dilation", layer.dilation(), (1, 1), pair(layer.dilation()))
            .unless("groups", layer.groups(), 1, layer.groups().to_string())
            .unless("bias", bias, true, py_bool(bias))
            .call("Conv2d"),
            ModuleType::Conv1d(layer) => Args::new(format!(
                "in_channels={}, out_channels={}, kernel_size={}",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ))
            .unless("stride", layer.stride(), 1, layer.stride().to_string())
            .unless(
                "padding",
                layer.is_same_padding(),
                false,
                "'same'".to_string(),
            )
            .unless(
                "padding",
                if layer.is_same_padding() {
                    0
                } else {
                    layer.padding()
                },
                0,
                layer.padding().to_string(),
            )
            .unless(
                "dilation",
                layer.dilation(),
                1,
                layer.dilation().to_string(),
            )
            .unless("groups", layer.groups(), 1, layer.groups().to_string())
            .unless("bias", bias, true, py_bool(bias))
            .call("Conv1d"),
            ModuleType::ConvTranspose2d(layer) => Args::new(format!(
                "in_channels={}, out_channels={}, kernel_size={}",
                layer.in_channels(),
                layer.out_channels(),
                pair(layer.kernel_size())
            ))
            .unless("stride", layer.stride(), (1, 1), pair(layer.stride()))
            .unless("padding", layer.padding(), (0, 0), pair(layer.padding()))
            .unless(
                "output_padding",
                layer.output_padding(),
                (0, 0),
                pair(layer.output_padding()),
            )
            .unless("dilation", layer.dilation(), (1, 1), pair(layer.dilation()))
            .unless("groups", layer.groups(), 1, layer.groups().to_string())
            .unless("bias", bias, true, py_bool(bias))
            .call("ConvTranspose2d"),
            ModuleType::ConvTranspose1d(layer) => Args::new(format!(
                "in_channels={}, out_channels={}, kernel_size={}",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ))
            .unless("stride", layer.stride(), 1, layer.stride().to_string())
            .unless("padding", layer.padding(), 0, layer.padding().to_string())
            .unless(
                "output_padding",
                layer.output_padding(),
                0,
                layer.output_padding().to_string(),
            )
            .unless(
                "dilation",
                layer.dilation(),
                1,
                layer.dilation().to_string(),
            )
            .unless("groups", layer.groups(), 1, layer.groups().to_string())
            .unless("bias", bias, true, py_bool(bias))
            .call("ConvTranspose1d"),
            ModuleType::Upsample(layer) => {
                let target = match (layer.size(), layer.scale_factor()) {
                    (Some(size), _) => format!("size={}", per_axis(size)),
                    (None, Some(factor)) => format!("scale_factor={}", per_axis(factor)),
                    (None, None) => String::new(),
                };
                let mode = mode_name(layer.mode());
                let args = Args(target)
                    .unless("mode", mode, "nearest", format!("\"{mode}\""))
                    .unless(
                        "align_corners",
                        layer.align_corners(),
                        false,
                        py_bool(layer.align_corners()),
                    );
                format!("Upsample({})", args.0.trim_start_matches(", "))
            }
            ModuleType::AdaptiveAvgPool2d(layer) => {
                format!(
                    "AdaptiveAvgPool2d(output_size={})",
                    pair(layer.output_size())
                )
            }
            ModuleType::AdaptiveMaxPool2d(layer) => {
                format!(
                    "AdaptiveMaxPool2d(output_size={})",
                    pair(layer.output_size())
                )
            }
            ModuleType::AdaptiveAvgPool1d(layer) => {
                format!("AdaptiveAvgPool1d(output_size={})", layer.output_size())
            }
            ModuleType::AdaptiveMaxPool1d(layer) => {
                format!("AdaptiveMaxPool1d(output_size={})", layer.output_size())
            }
            ModuleType::BatchNorm1d(layer) => {
                Args::new(format!("num_features={}", layer.num_features()))
                    .unless("eps", layer.eps(), 1e-5, float(layer.eps()))
                    .unless("momentum", layer.momentum(), 0.1, float(layer.momentum()))
                    .unless("affine", layer.affine(), true, py_bool(layer.affine()))
                    .call("BatchNorm1d")
            }
            ModuleType::BatchNorm2d(layer) => {
                Args::new(format!("num_features={}", layer.num_features()))
                    .unless("eps", layer.eps(), 1e-5, float(layer.eps()))
                    .unless("momentum", layer.momentum(), 0.1, float(layer.momentum()))
                    .unless("affine", layer.affine(), true, py_bool(layer.affine()))
                    .call("BatchNorm2d")
            }
            ModuleType::Dropout(layer) => format!("Dropout(p={})", float(layer.p())),
            ModuleType::Dropout2d(layer) => format!("Dropout2d(p={})", float(layer.p())),
            ModuleType::Embedding(layer) => Args::new(format!(
                "num_embeddings={}, embedding_dim={}",
                layer.num_embeddings(),
                layer.embedding_dim()
            ))
            .unless(
                "padding_idx",
                layer.padding_idx(),
                None,
                layer.padding_idx().map_or(String::new(), |i| i.to_string()),
            )
            .call("Embedding"),
            ModuleType::LayerNorm(layer) => {
                Args::new(format!("normalized_shape={:?}", layer.normalized_shape()))
                    .unless("eps", layer.eps(), 1e-5, float(layer.eps()))
                    .unless(
                        "elementwise_affine",
                        layer.elementwise_affine(),
                        true,
                        py_bool(layer.elementwise_affine()),
                    )
                    .call("LayerNorm")
            }
            ModuleType::RMSNorm(layer) => {
                Args::new(format!("normalized_shape={:?}", layer.normalized_shape()))
                    .unless("eps", layer.eps(), 1e-6, float(layer.eps()))
                    .unless(
                        "elementwise_affine",
                        layer.elementwise_affine(),
                        true,
                        py_bool(layer.elementwise_affine()),
                    )
                    .call("RMSNorm")
            }
            ModuleType::MultiheadAttention(layer) => Args::new(format!(
                "embed_dim={}, num_heads={}",
                layer.embed_dim(),
                layer.num_heads()
            ))
            .unless("bias", bias, true, py_bool(bias))
            .unless(
                "is_causal",
                layer.is_causal(),
                false,
                py_bool(layer.is_causal()),
            )
            .call("MultiheadAttention"),
            ModuleType::MaxPool2d(layer) => format!(
                "MaxPool2d(kernel_size={}, stride={}, padding={})",
                pair(layer.kernel_size()),
                pair(layer.stride()),
                pair(layer.padding())
            ),
            ModuleType::AvgPool2d(layer) => format!(
                "AvgPool2d(kernel_size={}, stride={}, padding={}, count_include_pad={})",
                pair(layer.kernel_size()),
                pair(layer.stride()),
                pair(layer.padding()),
                py_bool(layer.count_include_pad())
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
        })
    }

    /// Save module state to a file (basic implementation)
    #[pyo3(signature = (path, format=None))]
    fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        // Build a SerializedModel with metadata and engine state_dict
        let state = self.inner.get()?.as_module().state_dict();

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

    /// An independent copy of this layer: the same class and configuration,
    /// with every parameter and buffer in storage of its own.
    ///
    /// Training either one leaves the other alone, and an optimizer built over
    /// one never touches the other's parameters. Only the built-in layers can
    /// be copied this way; for any other model, build a second one and load
    /// the first one's weights with `load_state_dict(model.state_dict())`.
    fn __deepcopy__(slf: &Bound<'_, Self>, _memo: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let copy = slf.borrow().independent_copy(slf.py())?;
        rebuild_as(&slf.get_type(), copy)
    }

    /// Return a StateDict snapshot of this module
    fn state_dict(&self) -> PyResult<PyStateDict> {
        let state = self.inner.get()?.as_module().state_dict();
        Ok(crate::serialization::PyStateDict::from_engine(state))
    }

    /// Load `state` into this module: a `StateDict`, or any mapping from name
    /// to tensor.
    ///
    /// A state dict reads as a mapping, so `dict(state)` and a comprehension
    /// over `state.items()` are the natural way to change one -- casting its
    /// tensors, say -- and what they make has to load back. A mapping does not
    /// say which entries are buffers, so each name is filed where this module
    /// keeps it.
    #[pyo3(signature = (state, device=None))]
    fn load_state_dict(&mut self, state: &Bound<PyAny>, device: Option<&PyDevice>) -> PyResult<()> {
        let dev = device
            .map(|d| crate::device::ensure_available(d.device()))
            .transpose()?;
        let (given, built);
        let state_dict = match state.cast::<PyStateDict>() {
            Ok(state) => {
                given = state.borrow();
                PyStateDict::inner_ref(&given)
            }
            // A mapping is read before the module is borrowed for the write:
            // iterating one runs Python code, which may reach this module.
            Err(_) => {
                let buffers = self.inner.get()?.as_module().buffer_names();
                built = state_dict_from_mapping(state, &buffers)?;
                &built
            }
        };
        self.inner
            .get_mut()?
            .as_module_mut()
            .load_state_dict(state_dict, dev)
            .map_err(_convert_error)
    }
}

/// A state dict holding `mapping`'s tensors, each filed as a buffer if its name
/// is one of `buffers` and as a parameter otherwise.
fn state_dict_from_mapping(
    mapping: &Bound<PyAny>,
    buffers: &[String],
) -> PyResult<engine::serialization::StateDict> {
    let items = mapping.call_method0("items").map_err(|_| {
        PyTypeError::new_err(format!(
            "load_state_dict takes a StateDict or a mapping from name to tensor, not {}",
            type_name(mapping)
        ))
    })?;
    let mut state = engine::serialization::StateDict::new();
    for item in items.try_iter()? {
        let (name, value): (Bound<PyAny>, Bound<PyAny>) = item?.extract()?;
        let name: String = name.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "load_state_dict: every name must be a str, not {}",
                type_name(&name)
            ))
        })?;
        let tensor = value.cast::<PyTensor>().map_err(|_| {
            PyTypeError::new_err(format!(
                "load_state_dict: {name:?} holds a {}, not a Tensor",
                type_name(&value)
            ))
        })?;
        let tensor = tensor.borrow();
        let added = if buffers.contains(&name) {
            state.add_buffer(name, tensor.tensor())
        } else {
            state.add_parameter(name, tensor.tensor())
        };
        added.map_err(_convert_error)?;
    }
    Ok(state)
}

fn type_name(value: &Bound<PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

impl ModuleType {
    /// A copy of the layer that shares its tensors, or `None` for a
    /// `Sequential` holding a child that cannot be copied.
    fn try_clone(&self) -> Option<ModuleType> {
        Some(match self {
            ModuleType::Sequential(model) => ModuleType::Sequential(Box::new(model.try_clone()?)),
            ModuleType::DenseLayer(layer) => ModuleType::DenseLayer(layer.clone()),
            ModuleType::ReLU(layer) => ModuleType::ReLU(layer.clone()),
            ModuleType::Sigmoid(layer) => ModuleType::Sigmoid(layer.clone()),
            ModuleType::Tanh(layer) => ModuleType::Tanh(layer.clone()),
            ModuleType::Softmax(layer) => ModuleType::Softmax(layer.clone()),
            ModuleType::LeakyReLU(layer) => ModuleType::LeakyReLU(layer.clone()),
            ModuleType::Elu(layer) => ModuleType::Elu(layer.clone()),
            ModuleType::Gelu(layer) => ModuleType::Gelu(layer.clone()),
            ModuleType::Conv2d(layer) => ModuleType::Conv2d(layer.clone()),
            ModuleType::BatchNorm1d(layer) => ModuleType::BatchNorm1d(layer.clone()),
            ModuleType::BatchNorm2d(layer) => ModuleType::BatchNorm2d(layer.clone()),
            ModuleType::Dropout(layer) => ModuleType::Dropout(layer.clone()),
            ModuleType::Dropout2d(layer) => ModuleType::Dropout2d(layer.clone()),
            ModuleType::Embedding(layer) => ModuleType::Embedding(layer.clone()),
            ModuleType::MultiheadAttention(layer) => ModuleType::MultiheadAttention(layer.clone()),
            ModuleType::LayerNorm(layer) => ModuleType::LayerNorm(layer.clone()),
            ModuleType::RMSNorm(layer) => ModuleType::RMSNorm(layer.clone()),
            ModuleType::MaxPool2d(layer) => ModuleType::MaxPool2d(layer.clone()),
            ModuleType::AvgPool2d(layer) => ModuleType::AvgPool2d(layer.clone()),
            ModuleType::Recurrent(layer) => ModuleType::Recurrent(layer.clone()),
            ModuleType::Conv1d(layer) => ModuleType::Conv1d(layer.clone()),
            ModuleType::MaxPool1d(layer) => ModuleType::MaxPool1d(layer.clone()),
            ModuleType::AvgPool1d(layer) => ModuleType::AvgPool1d(layer.clone()),
            ModuleType::ConvTranspose2d(layer) => ModuleType::ConvTranspose2d(layer.clone()),
            ModuleType::ConvTranspose1d(layer) => ModuleType::ConvTranspose1d(layer.clone()),
            ModuleType::AdaptiveAvgPool2d(layer) => ModuleType::AdaptiveAvgPool2d(layer.clone()),
            ModuleType::AdaptiveMaxPool2d(layer) => ModuleType::AdaptiveMaxPool2d(layer.clone()),
            ModuleType::AdaptiveAvgPool1d(layer) => ModuleType::AdaptiveAvgPool1d(layer.clone()),
            ModuleType::AdaptiveMaxPool1d(layer) => ModuleType::AdaptiveMaxPool1d(layer.clone()),
            ModuleType::Upsample(layer) => ModuleType::Upsample(layer.clone()),
        })
    }
}

impl PyModule {
    /// This module with every parameter and buffer copied into storage of its
    /// own, under a fresh identity.
    ///
    /// Cloning the layer alone shares every tensor's storage and id with the
    /// original, and a parameter is updated in place: an optimizer stepping
    /// the copy would move the original's weights too, and its state, keyed
    /// by tensor id, would treat the two as one parameter. The other tensors a
    /// layer holds are only ever read, so sharing them is safe.
    ///
    /// A `Sequential` is copied child by child, each into a module object of
    /// its own class, so the copy can be indexed like the original and each
    /// of its parts belongs to it alone.
    fn independent_copy(&self, py: Python<'_>) -> PyResult<Self> {
        if let ModuleType::Sequential(_) = self.inner.get()? {
            let mut copy = PyModule::from_sequential(Sequential::new());
            let names = self.child_names()?;
            let mut layers = Vec::new();
            for child in self.children(py)? {
                let child = child.bind(py);
                let original = child.cast::<PyModule>()?;
                let copied = original.borrow().independent_copy(py)?;
                let copied = rebuild_as(&child.get_type(), copied)?;
                layers.push(PyModule::adopt_into(
                    copied.bind(py).cast::<PyModule>()?,
                    &copy,
                )?);
            }
            // Named as the original's layers are, so the copy's parameter
            // keys match and a state dict moves between the two.
            if let ModuleType::Sequential(model) = copy.inner.get_mut()? {
                for (name, layer) in names.iter().zip(layers) {
                    if name.parse::<u64>().is_ok() {
                        model.add_layer(layer);
                    } else {
                        model.add_named_layer(name, layer).map_err(_convert_error)?;
                    }
                }
            }
            copy.inner.0.set_training(self.inner.0.is_training());
            return Ok(copy);
        }
        fn independent(tensor: &engine::tensor::Tensor) -> engine::tensor::Tensor {
            engine::tensor::Tensor::new(
                std::sync::Arc::new(tensor.data().clone_data()),
                tensor.shape().clone(),
                tensor.dtype(),
                tensor.device(),
                false,
            )
            // Set after the tensor exists, so a copy made inside `no_grad` --
            // where a snapshot usually is -- keeps its parameters trainable.
            .requires_grad_(tensor.requires_grad())
        }
        let mut inner = self.inner.get()?.try_clone().ok_or_else(|| {
            PyTypeError::new_err("this Sequential holds a layer that cannot be copied")
        })?;
        for tensor in inner.as_layer_mut().parameters_mut() {
            *tensor = independent(tensor);
        }
        for tensor in inner.as_layer_mut().buffers_mut() {
            *tensor = independent(tensor);
        }
        let copy = Self {
            inner: SharedModule::new(inner),
        };
        copy.inner.0.set_training(self.inner.0.is_training());
        Ok(copy)
    }

    pub fn from_dense_layer(dense_layer: DenseLayer) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::DenseLayer(Box::new(dense_layer))),
        }
    }

    pub fn from_relu(relu: ReLU) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::ReLU(Box::new(relu))),
        }
    }

    pub fn from_sigmoid(sigmoid: Sigmoid) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Sigmoid(Box::new(sigmoid))),
        }
    }

    pub fn from_tanh(tanh: Tanh) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Tanh(Box::new(tanh))),
        }
    }

    pub fn from_softmax(softmax: Softmax) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Softmax(Box::new(softmax))),
        }
    }

    pub fn from_leaky_relu(leaky_relu: LeakyReLU) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::LeakyReLU(Box::new(leaky_relu))),
        }
    }

    pub fn from_elu(elu: ELU) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Elu(Box::new(elu))),
        }
    }

    pub fn from_gelu(gelu: GELU) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Gelu(Box::new(gelu))),
        }
    }

    pub fn from_sequential(sequential: Sequential) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Sequential(Box::new(sequential))),
        }
    }

    pub fn from_conv2d(conv2d: Conv2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Conv2d(Box::new(conv2d))),
        }
    }

    pub fn from_upsample(layer: Upsample) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Upsample(Box::new(layer))),
        }
    }

    pub fn from_adaptive_avg_pool2d(layer: AdaptiveAvgPool2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AdaptiveAvgPool2d(Box::new(layer))),
        }
    }

    pub fn from_adaptive_max_pool2d(layer: AdaptiveMaxPool2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AdaptiveMaxPool2d(Box::new(layer))),
        }
    }

    pub fn from_adaptive_avg_pool1d(layer: AdaptiveAvgPool1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AdaptiveAvgPool1d(Box::new(layer))),
        }
    }

    pub fn from_adaptive_max_pool1d(layer: AdaptiveMaxPool1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AdaptiveMaxPool1d(Box::new(layer))),
        }
    }

    pub fn from_conv_transpose2d(layer: ConvTranspose2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::ConvTranspose2d(Box::new(layer))),
        }
    }

    pub fn from_conv_transpose1d(layer: ConvTranspose1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::ConvTranspose1d(Box::new(layer))),
        }
    }

    pub fn from_max_pool2d(max_pool2d: MaxPool2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::MaxPool2d(Box::new(max_pool2d))),
        }
    }

    pub fn from_avg_pool2d(avg_pool2d: AvgPool2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AvgPool2d(Box::new(avg_pool2d))),
        }
    }

    pub fn from_batch_norm1d(batch_norm1d: BatchNorm1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::BatchNorm1d(Box::new(batch_norm1d))),
        }
    }

    pub fn from_batch_norm2d(batch_norm2d: BatchNorm2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::BatchNorm2d(Box::new(batch_norm2d))),
        }
    }

    pub fn from_dropout(dropout: Dropout) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Dropout(Box::new(dropout))),
        }
    }

    pub fn from_dropout2d(dropout: Dropout2d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Dropout2d(Box::new(dropout))),
        }
    }

    pub fn from_embedding(embedding: Embedding) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Embedding(Box::new(embedding))),
        }
    }

    pub fn from_layer_norm(layer_norm: LayerNorm) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::LayerNorm(Box::new(layer_norm))),
        }
    }

    pub fn from_rms_norm(rms_norm: RMSNorm) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::RMSNorm(Box::new(rms_norm))),
        }
    }

    pub fn from_conv1d(conv1d: Conv1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Conv1d(Box::new(conv1d))),
        }
    }

    pub fn from_max_pool1d(layer: MaxPool1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::MaxPool1d(Box::new(layer))),
        }
    }

    pub fn from_avg_pool1d(layer: AvgPool1d) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::AvgPool1d(Box::new(layer))),
        }
    }

    pub fn from_recurrent(recurrent: Recurrent) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::Recurrent(Box::new(recurrent))),
        }
    }

    pub fn from_multihead_attention(mha: MultiheadAttention) -> Self {
        Self {
            inner: SharedModule::new(ModuleType::MultiheadAttention(Box::new(mha))),
        }
    }

    /// A `Sequential` as the modules it holds, one per line, each under the
    /// index that reaches it.
    fn sequential_repr(&self) -> PyResult<String> {
        Python::attach(|py| {
            let children = self.children(py)?;
            if children.is_empty() {
                return Ok("Sequential()".to_string());
            }
            let names = self.child_names()?;
            let mut text = String::from("Sequential(\n");
            for (name, child) in names.iter().zip(children.iter()) {
                let shown = child.bind(py).repr()?.to_string().replace('\n', "\n  ");
                text.push_str(&format!("  ({name}): {shown}\n"));
            }
            text.push(')');
            Ok(text)
        })
    }

    /// `module`'s layer, to be held by `parent`, a `Sequential`.
    pub(crate) fn adopt_into(
        module: &Bound<'_, PyModule>,
        parent: &PyModule,
    ) -> PyResult<Box<dyn Layer>> {
        let object = module.clone().into_any().unbind();
        module.borrow().inner.adopt_into(object, &parent.inner)
    }

    /// The modules a `Sequential` holds, as the Python objects they were
    /// added as.
    pub(crate) fn children(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner.check_idle()?;
        Ok(self
            .inner
            .0
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|child| child.object.clone_ref(py))
            .collect())
    }

    /// Append `children`, which `adopt_into` gave for this `Sequential`.
    pub(crate) fn push_children(&mut self, children: Vec<Box<dyn Layer>>) -> PyResult<()> {
        if let ModuleType::Sequential(model) = self.inner.get_mut()? {
            for child in children {
                model.add_layer(child);
            }
        }
        Ok(())
    }

    /// The names of a `Sequential`'s layers, in order; empty for any other
    /// module.
    pub(crate) fn child_names(&self) -> PyResult<Vec<String>> {
        Ok(match self.inner.get()? {
            ModuleType::Sequential(model) => model.names().to_vec(),
            _ => Vec::new(),
        })
    }
}

/// DenseLayer (fully connected) layer
#[pyclass(name = "DenseLayer", module = "minitensor.nn", extends = PyModule)]
pub struct PyDenseLayer;

#[pymethods]
impl PyDenseLayer {
    /// Create a new dense layer
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=None, device=None, dtype=None))]
    fn new(
        in_features: crate::Size,
        out_features: crate::Size,
        bias: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let in_features = in_features.get();
        let out_features = out_features.get();
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
        if let ModuleType::DenseLayer(layer) = module.inner.get()? {
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
        if let ModuleType::DenseLayer(layer) = module.inner.get()? {
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
        if let ModuleType::DenseLayer(layer) = module.inner.get()? {
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
        if let ModuleType::DenseLayer(layer) = module.inner.get()? {
            Ok(layer.bias().map(|b| PyTensor::from_tensor(b.clone())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// ReLU activation layer
#[pyclass(name = "ReLU", module = "minitensor.nn", extends = PyModule)]
pub struct PyReLU;

/// `module` as an instance of `class`, which must be `Module` or one of the
/// built-in layers: those are marker types over `Module`, so knowing the class
/// is all it takes to build one around an existing layer.
fn rebuild_as(class: &Bound<'_, pyo3::types::PyType>, module: PyModule) -> PyResult<Py<PyAny>> {
    let py = class.py();
    if class.is(py.get_type::<PyModule>()) {
        return Ok(Py::new(py, module)?.into_any());
    }
    macro_rules! built_in {
        ($($ty:ident),+ $(,)?) => {
            $(
                if class.is(py.get_type::<$ty>()) {
                    let init = PyClassInitializer::from(module).add_subclass($ty);
                    return Ok(Py::new(py, init)?.into_any());
                }
            )+
        };
    }
    built_in!(
        PyDenseLayer,
        PyReLU,
        PySigmoid,
        PyTanh,
        PySoftmax,
        PyLeakyReLU,
        PyELU,
        PyGELU,
        PyDropout,
        PyDropout2d,
        PyConv1d,
        PyConv2d,
        PyConvTranspose1d,
        PyConvTranspose2d,
        PyMaxPool1d,
        PyAvgPool1d,
        PyMaxPool2d,
        PyAvgPool2d,
        PyUpsample,
        PyAdaptiveAvgPool1d,
        PyAdaptiveAvgPool2d,
        PyAdaptiveMaxPool1d,
        PyAdaptiveMaxPool2d,
        PyBatchNorm1d,
        PyBatchNorm2d,
        PyEmbedding,
        PyLayerNorm,
        PyRMSNorm,
        PyMultiheadAttention,
        PySequential,
        PyLSTM,
        PyGRU,
    );
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "copy.deepcopy copies the built-in layers, and {} is not one; build a \
         second instance and load this one's weights into it with \
         load_state_dict(model.state_dict())",
        class.name()?
    )))
}

#[cfg(test)]
mod shared_module_tests {
    use super::*;

    fn relu() -> SharedModule {
        SharedModule::new(ModuleType::ReLU(Box::new(ReLU::new())))
    }

    fn sequential() -> SharedModule {
        SharedModule::new(ModuleType::Sequential(Box::new(Sequential::new())))
    }

    /// While a model is in a forward pass -- which may release the GIL --
    /// nothing else may reach any module in it: not the modules' own Python
    /// objects, and not a second forward pass. Threads would make this a
    /// race to observe; the lock is what decides it, so hold it and look.
    #[test]
    fn a_model_in_a_forward_pass_is_closed_to_everything_else() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut parent = sequential();
            let mut child = relu();
            let adopted = child.adopt_into(py.None(), &parent).unwrap();
            if let ModuleType::Sequential(model) = parent.get_mut().unwrap() {
                model.add_layer(adopted);
            }

            let other = relu();
            parent
                .forward(|_| {
                    assert!(child.get().is_err());
                    assert!(child.get_mut().is_err());
                    assert!(child.forward(|_| ()).is_err());
                    assert!(other.adopt_into(py.None(), &child).is_err());
                    // A model the lock does not cover is untouched.
                    assert!(other.get().is_ok());
                })
                .unwrap();

            assert!(child.get().is_ok());
            assert!(child.forward(|_| ()).is_ok());
        });
    }

    #[test]
    fn a_dropped_parent_gives_its_children_locks_of_their_own() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut first = relu();
            let second = relu();
            {
                let parent = sequential();
                drop(first.adopt_into(py.None(), &parent).unwrap());
                drop(second.adopt_into(py.None(), &parent).unwrap());
            }
            first.forward(|_| assert!(second.get().is_ok())).unwrap();
            assert!(second.adopt_into(py.None(), &relu()).is_ok());
        });
    }
}
