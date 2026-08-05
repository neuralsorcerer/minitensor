// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Weight initialization.
//!
//! Six of these already existed as tensor constructors -- `mt.xavier_uniform`,
//! `mt.he_normal` and so on, each with a `_like` variant taking a reference
//! tensor instead of a shape. What was missing was the namespace a PyTorch user
//! looks in, `calculate_fan_in_and_fan_out`, and the Kaiming/Glorot spellings
//! of names this library had picked a side on.
//!
//! **The `requires_grad` default differs from the top-level constructors, on
//! purpose.** `mt.xavier_uniform(shape)` sits beside `mt.zeros` and `mt.randn`
//! as a way to make a tensor, and defaults to `false` like they do.
//! `nn.init.xavier_uniform(shape)` exists to make a *parameter* -- the reason
//! it was added was `plugins.CustomLayer`, whose parameters are plain tensors
//! -- and a parameter created without `requires_grad` does not train, silently.
//! So this namespace defaults to `true`. Both take the argument explicitly.
//!
//! These are factories, not PyTorch's in-place `xavier_uniform_(tensor)`: the
//! engine's initializers build a tensor from a shape, which is also how every
//! layer in `nn` creates its parameters, and there is no in-place fill to wrap.
//! To re-initialize an existing parameter, build a new tensor and assign it.
//!
//! Getting `fan_in` and `fan_out` the wrong way round is the usual way a
//! hand-rolled scheme goes wrong, because a weight here is stored
//! `[out_features, in_features]` and the fan the formulas want is the trailing
//! dimension. That is what `calculate_fan_in_and_fan_out` is exposed for.

use crate::device::{PyDevice, resolve_device};
use crate::dtype::parse_dtype;
use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::nn::init;
use engine::tensor::Shape;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule as Pyo3Module;

/// Shared argument handling: shape, dtype and device, with the defaults every
/// one of these takes.
fn prepare(
    shape: Vec<isize>,
    dtype: Option<&str>,
    device: Option<PyDevice>,
) -> PyResult<(Shape, engine::tensor::DataType, engine::device::Device)> {
    let mut dims = Vec::with_capacity(shape.len());
    for extent in shape {
        if extent < 0 {
            return Err(PyValueError::new_err(format!(
                "shape must not contain negative dimensions, got {extent}"
            )));
        }
        dims.push(extent as usize);
    }
    let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
    let device = resolve_device(device.as_ref())?;
    Ok((Shape::new(dims), dtype, device))
}

/// Only the float dtypes have a meaningful distribution to draw from; the
/// engine's random initializers reject the rest, but they reject it deep enough
/// that the message does not mention initialization.
fn require_float(dtype: engine::tensor::DataType, name: &str) -> PyResult<()> {
    if dtype.is_float() {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} draws from a continuous distribution and needs a float dtype, got {dtype:?}"
    )))
}

macro_rules! fan_based {
    ($name:ident, $engine_fn:ident, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=true))]
        fn $name(
            shape: Vec<isize>,
            dtype: Option<&str>,
            device: Option<PyDevice>,
            requires_grad: bool,
        ) -> PyResult<PyTensor> {
            let (shape, dtype, device) = prepare(shape, dtype, device)?;
            require_float(dtype, stringify!($name))?;
            let tensor =
                init::$engine_fn(shape, dtype, device, requires_grad).map_err(_convert_error)?;
            Ok(PyTensor::from_tensor(tensor))
        }
    };
}

fan_based!(
    xavier_uniform,
    xavier_uniform_init,
    "Uniform over +/- sqrt(6 / (fan_in + fan_out)). Glorot & Bengio (2010)."
);
fan_based!(
    xavier_normal,
    xavier_normal_init,
    "Normal with std sqrt(2 / (fan_in + fan_out)). Glorot & Bengio (2010)."
);
fan_based!(
    he_uniform,
    he_uniform_init,
    "Uniform over +/- sqrt(6 / fan_in). He et al. (2015); for ReLU networks."
);
fan_based!(
    he_normal,
    he_normal_init,
    "Normal with std sqrt(2 / fan_in). He et al. (2015); for ReLU networks."
);
fan_based!(
    lecun_uniform,
    lecun_uniform_init,
    "Uniform over +/- sqrt(3 / fan_in). LeCun et al. (1998)."
);
fan_based!(
    lecun_normal,
    lecun_normal_init,
    "Normal with std sqrt(1 / fan_in). LeCun et al. (1998); for SELU networks."
);

/// A tensor of zeros.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None, device=None, requires_grad=true))]
fn zeros(
    shape: Vec<isize>,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    constant(shape, 0.0, dtype, device, requires_grad)
}

/// A tensor of ones.
#[pyfunction]
#[pyo3(signature = (shape, dtype=None, device=None, requires_grad=true))]
fn ones(
    shape: Vec<isize>,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    constant(shape, 1.0, dtype, device, requires_grad)
}

/// A tensor filled with `value`.
#[pyfunction]
#[pyo3(signature = (shape, value, dtype=None, device=None, requires_grad=true))]
fn constant(
    shape: Vec<isize>,
    value: f64,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = prepare(shape, dtype, device)?;
    let tensor =
        init::init_constant(shape, value, dtype, device, requires_grad).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(tensor))
}

/// Uniform over `[a, b)`.
#[pyfunction]
#[pyo3(signature = (shape, a=0.0, b=1.0, dtype=None, device=None, requires_grad=true))]
fn uniform(
    shape: Vec<isize>,
    a: f64,
    b: f64,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = prepare(shape, dtype, device)?;
    require_float(dtype, "uniform")?;
    // Both bounds finite and ordered. A NaN bound reaches the sampler as a
    // range it cannot construct, and an empty range (a >= b) silently yields
    // a constant tensor.
    if !a.is_finite() || !b.is_finite() || a >= b {
        return Err(PyValueError::new_err(format!(
            "uniform needs finite bounds with a < b, got a={a} and b={b}"
        )));
    }
    let tensor =
        init::init_uniform(shape, a, b, dtype, device, requires_grad).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(tensor))
}

/// Normal with the given mean and standard deviation.
#[pyfunction]
#[pyo3(signature = (shape, mean=0.0, std=1.0, dtype=None, device=None, requires_grad=true))]
fn normal(
    shape: Vec<isize>,
    mean: f64,
    std: f64,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = prepare(shape, dtype, device)?;
    require_float(dtype, "normal")?;
    let tensor = init::init_normal(shape, mean, std, dtype, device, requires_grad)
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(tensor))
}

/// Normal, resampled until every value lies in `[lower, upper]`.
///
/// The bounds are given in the same units as the values, not in standard
/// deviations, and default to two deviations either side of the mean.
#[pyfunction]
#[pyo3(signature = (
    shape, mean=0.0, std=1.0, lower=None, upper=None,
    dtype=None, device=None, requires_grad=true
))]
#[allow(clippy::too_many_arguments)]
fn truncated_normal(
    shape: Vec<isize>,
    mean: f64,
    std: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    dtype: Option<&str>,
    device: Option<PyDevice>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = prepare(shape, dtype, device)?;
    require_float(dtype, "truncated_normal")?;
    let lower = lower.unwrap_or(mean - 2.0 * std);
    let upper = upper.unwrap_or(mean + 2.0 * std);
    let tensor =
        init::truncated_normal_init(shape, mean, std, lower, upper, dtype, device, requires_grad)
            .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(tensor))
}

/// The `(fan_in, fan_out)` the schemes above derive their scale from.
///
/// Exposed because a caller writing their own scheme needs the same numbers,
/// and deriving them independently is how a hand-rolled initializer ends up
/// transposed: a weight is stored `[out_features, in_features]`, so `fan_in` is
/// the trailing dimension, and a convolution weight's fans are scaled by its
/// receptive field.
#[pyfunction]
fn calculate_fan_in_and_fan_out(shape: Vec<isize>) -> PyResult<(usize, usize)> {
    let (shape, _, _) = prepare(shape, None, None)?;
    init::calculate_fan_in_fan_out(&shape).map_err(_convert_error)
}

pub fn register(py: Python, parent: &Bound<Pyo3Module>) -> PyResult<()> {
    let module = Pyo3Module::new(py, "init")?;
    module.setattr(
        "__doc__",
        "Weight initialization schemes for parameters you create yourself.",
    )?;
    module.add_function(wrap_pyfunction!(zeros, &module)?)?;
    module.add_function(wrap_pyfunction!(ones, &module)?)?;
    module.add_function(wrap_pyfunction!(constant, &module)?)?;
    module.add_function(wrap_pyfunction!(uniform, &module)?)?;
    module.add_function(wrap_pyfunction!(normal, &module)?)?;
    module.add_function(wrap_pyfunction!(truncated_normal, &module)?)?;
    module.add_function(wrap_pyfunction!(xavier_uniform, &module)?)?;
    module.add_function(wrap_pyfunction!(xavier_normal, &module)?)?;
    module.add_function(wrap_pyfunction!(he_uniform, &module)?)?;
    module.add_function(wrap_pyfunction!(he_normal, &module)?)?;
    module.add_function(wrap_pyfunction!(lecun_uniform, &module)?)?;
    module.add_function(wrap_pyfunction!(lecun_normal, &module)?)?;
    module.add_function(wrap_pyfunction!(calculate_fan_in_and_fan_out, &module)?)?;

    // He and Kaiming are the same person and the same paper; both spellings
    // are in circulation, and a user who reaches for the other one should not
    // have to discover that this library picked a side.
    module.add("kaiming_uniform", module.getattr("he_uniform")?)?;
    module.add("kaiming_normal", module.getattr("he_normal")?)?;
    // Glorot likewise, for Xavier Glorot.
    module.add("glorot_uniform", module.getattr("xavier_uniform")?)?;
    module.add("glorot_normal", module.getattr("xavier_normal")?)?;

    parent.add_submodule(&module)?;
    Ok(())
}
