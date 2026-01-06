// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::dtype;
use crate::error::_convert_error;
use crate::serialization::PyStateDict;
use crate::tensor::PyTensor;
use engine::Device;
use engine::nn::{
    BCELoss, CrossEntropyLoss, DenseLayer, FocalLoss, HuberLoss, Layer, LogCoshLoss, MAELoss,
    MSELoss, ReLU, Sequential, Sigmoid, SmoothL1Loss, Softmax, Tanh,
    activation::{ELU, GELU, LeakyReLU},
    conv::Conv2d,
    dropout::{Dropout, Dropout2d},
    normalization::{BatchNorm1d, BatchNorm2d},
    utils::{LayerUtils, SequentialUtils},
};
use engine::operations::batch_norm as batch_norm_op;
use engine::operations::conv2d as conv2d_op;
use engine::operations::linalg::matmul as matmul_op;
use engine::operations::loss::cross_entropy as cross_entropy_op;
use engine::serialization::{ModelMetadata, ModelSerializer, SerializationFormat, SerializedModel};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule as Pyo3Module};

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

#[pyfunction]
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

    let weight_t = weight_tensor
        .tensor()
        .transpose(0, 1)
        .map_err(_convert_error)?;
    let mut output = matmul_op(input_tensor.tensor(), &weight_t).map_err(_convert_error)?;

    let bias_tensor = borrow_optional_tensor(bias)?;
    if let Some(bias_ref) = bias_tensor {
        output = output.add(bias_ref.tensor()).map_err(_convert_error)?;
    }

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

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
fn conv2d(
    input: &Bound<PyAny>,
    weight: &Bound<PyAny>,
    bias: Option<&Bound<PyAny>>,
    stride: Option<&Bound<PyAny>>,
    padding: Option<&Bound<PyAny>>,
) -> PyResult<PyTensor> {
    let input_tensor = borrow_tensor(input)?;
    let weight_tensor = borrow_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    let stride = parse_pair_arg("stride", stride, (1, 1))?;
    let padding = parse_pair_arg("padding", padding, (0, 0))?;
    let result = conv2d_op(
        input_tensor.tensor(),
        weight_tensor.tensor(),
        bias_tensor.as_ref().map(|b| b.tensor()),
        stride,
        padding,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=true, momentum=0.1, eps=1e-5))]
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

    let ndim = input_tensor.tensor().ndim() as isize;
    let axis = if dim < 0 { ndim + dim } else { dim };
    if axis < 0 || axis as usize >= ndim as usize {
        return Err(PyIndexError::new_err("dim out of range"));
    }
    let result = cross_entropy_op(
        input_tensor.tensor(),
        target_tensor.tensor(),
        reduction,
        axis as usize,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

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

#[pyfunction(name = "mse_loss")]
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

#[pyfunction(name = "smooth_l1_loss")]
fn smooth_l1_loss_functional(
    input: &Bound<PyAny>,
    target: &Bound<PyAny>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let prediction = borrow_tensor(input)?;
    let target_tensor = borrow_tensor(target)?;
    let reduction = reduction.unwrap_or("mean");
    let loss = SmoothL1Loss::new(reduction);
    let result = loss
        .forward(prediction.tensor(), target_tensor.tensor())
        .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

#[pyfunction(name = "log_cosh_loss")]
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

/// Base class for neural network modules
#[pyclass(name = "Module", subclass)]
pub struct PyModule {
    // This will be a trait object in practice
    // For now, we'll use an enum to handle different layer types
    inner: ModuleType,
}

enum ModuleType {
    DenseLayer(DenseLayer),
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    Softmax(Softmax),
    LeakyReLU(LeakyReLU),
    ELU(ELU),
    GELU(GELU),
    Sequential(Sequential),
    Conv2d(Conv2d),
    BatchNorm1d(BatchNorm1d),
    BatchNorm2d(BatchNorm2d),
    Dropout(Dropout),
    Dropout2d(Dropout2d),
}

#[pymethods]
impl PyModule {
    /// Forward pass through the module
    fn forward(&mut self, input: &Bound<PyAny>) -> PyResult<PyTensor> {
        let input_tensor = borrow_tensor(input)?;
        let result = match &mut self.inner {
            ModuleType::DenseLayer(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::ReLU(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Sigmoid(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Tanh(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Softmax(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::LeakyReLU(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::ELU(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::GELU(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Sequential(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Conv2d(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::BatchNorm1d(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::BatchNorm2d(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Dropout(layer) => layer.forward(input_tensor.tensor()),
            ModuleType::Dropout2d(layer) => layer.forward(input_tensor.tensor()),
        }
        .map_err(_convert_error)?;

        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&mut self, input: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(input)
    }

    /// Get all parameters of the module
    fn parameters(&self) -> Vec<PyTensor> {
        let params = match &self.inner {
            ModuleType::DenseLayer(layer) => layer.parameters(),
            ModuleType::ReLU(layer) => layer.parameters(),
            ModuleType::Sigmoid(layer) => layer.parameters(),
            ModuleType::Tanh(layer) => layer.parameters(),
            ModuleType::Softmax(layer) => layer.parameters(),
            ModuleType::LeakyReLU(layer) => layer.parameters(),
            ModuleType::ELU(layer) => layer.parameters(),
            ModuleType::GELU(layer) => layer.parameters(),
            ModuleType::Sequential(layer) => layer.parameters(),
            ModuleType::Conv2d(layer) => layer.parameters(),
            ModuleType::BatchNorm1d(layer) => layer.parameters(),
            ModuleType::BatchNorm2d(layer) => layer.parameters(),
            ModuleType::Dropout(layer) => layer.parameters(),
            ModuleType::Dropout2d(layer) => layer.parameters(),
        };

        params
            .into_iter()
            .map(|tensor| PyTensor::from_tensor(tensor.clone()))
            .collect()
    }

    /// Set module to training mode
    fn train(&mut self) {
        match &mut self.inner {
            ModuleType::DenseLayer(layer) => layer.train(),
            ModuleType::ReLU(layer) => layer.train(),
            ModuleType::Sigmoid(layer) => layer.train(),
            ModuleType::Tanh(layer) => layer.train(),
            ModuleType::Softmax(layer) => layer.train(),
            ModuleType::LeakyReLU(layer) => layer.train(),
            ModuleType::ELU(layer) => layer.train(),
            ModuleType::GELU(layer) => layer.train(),
            ModuleType::Sequential(layer) => layer.train(),
            ModuleType::Conv2d(layer) => layer.train(),
            ModuleType::BatchNorm1d(layer) => layer.train(),
            ModuleType::BatchNorm2d(layer) => layer.train(),
            ModuleType::Dropout(layer) => layer.train(),
            ModuleType::Dropout2d(layer) => layer.train(),
        }
    }

    /// Set module to evaluation mode
    fn eval(&mut self) {
        match &mut self.inner {
            ModuleType::DenseLayer(layer) => layer.eval(),
            ModuleType::ReLU(layer) => layer.eval(),
            ModuleType::Sigmoid(layer) => layer.eval(),
            ModuleType::Tanh(layer) => layer.eval(),
            ModuleType::Softmax(layer) => layer.eval(),
            ModuleType::LeakyReLU(layer) => layer.eval(),
            ModuleType::ELU(layer) => layer.eval(),
            ModuleType::GELU(layer) => layer.eval(),
            ModuleType::Sequential(layer) => layer.eval(),
            ModuleType::Conv2d(layer) => layer.eval(),
            ModuleType::BatchNorm1d(layer) => layer.eval(),
            ModuleType::BatchNorm2d(layer) => layer.eval(),
            ModuleType::Dropout(layer) => layer.eval(),
            ModuleType::Dropout2d(layer) => layer.eval(),
        }
    }

    /// Get number of parameters
    fn num_parameters(&self) -> usize {
        match &self.inner {
            ModuleType::DenseLayer(layer) => layer.num_parameters(),
            ModuleType::ReLU(layer) => layer.num_parameters(),
            ModuleType::Sigmoid(layer) => layer.num_parameters(),
            ModuleType::Tanh(layer) => layer.num_parameters(),
            ModuleType::Softmax(layer) => layer.num_parameters(),
            ModuleType::LeakyReLU(layer) => layer.num_parameters(),
            ModuleType::ELU(layer) => layer.num_parameters(),
            ModuleType::GELU(layer) => layer.num_parameters(),
            ModuleType::Sequential(layer) => layer.num_parameters(),
            ModuleType::Conv2d(layer) => layer.num_parameters(),
            ModuleType::BatchNorm1d(layer) => layer.num_parameters(),
            ModuleType::BatchNorm2d(layer) => layer.num_parameters(),
            ModuleType::Dropout(layer) => layer.num_parameters(),
            ModuleType::Dropout2d(layer) => layer.num_parameters(),
        }
    }

    /// Get detailed parameter statistics
    fn parameter_stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let layer: &dyn Layer = match &self.inner {
            ModuleType::DenseLayer(layer) => layer,
            ModuleType::ReLU(layer) => layer,
            ModuleType::Sigmoid(layer) => layer,
            ModuleType::Tanh(layer) => layer,
            ModuleType::Softmax(layer) => layer,
            ModuleType::LeakyReLU(layer) => layer,
            ModuleType::ELU(layer) => layer,
            ModuleType::GELU(layer) => layer,
            ModuleType::Sequential(layer) => layer,
            ModuleType::Conv2d(layer) => layer,
            ModuleType::BatchNorm1d(layer) => layer,
            ModuleType::BatchNorm2d(layer) => layer,
            ModuleType::Dropout(layer) => layer,
            ModuleType::Dropout2d(layer) => layer,
        };
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
        let layer: &dyn Layer = match &self.inner {
            ModuleType::DenseLayer(layer) => layer,
            ModuleType::ReLU(layer) => layer,
            ModuleType::Sigmoid(layer) => layer,
            ModuleType::Tanh(layer) => layer,
            ModuleType::Softmax(layer) => layer,
            ModuleType::LeakyReLU(layer) => layer,
            ModuleType::ELU(layer) => layer,
            ModuleType::GELU(layer) => layer,
            ModuleType::Sequential(layer) => layer,
            ModuleType::Conv2d(layer) => layer,
            ModuleType::BatchNorm1d(layer) => layer,
            ModuleType::BatchNorm2d(layer) => layer,
            ModuleType::Dropout(layer) => layer,
            ModuleType::Dropout2d(layer) => layer,
        };
        let usage = LayerUtils::memory_usage(layer);
        let dict = PyDict::new(py);
        dict.set_item("total_bytes", usage.total_bytes)?;
        let dtype_dict = PyDict::new(py);
        for (dtype, bytes) in usage.bytes_by_dtype {
            dtype_dict.set_item(format!("{:?}", dtype), bytes)?;
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
                let layer: &dyn Layer = match &self.inner {
                    ModuleType::DenseLayer(layer) => layer,
                    ModuleType::ReLU(layer) => layer,
                    ModuleType::Sigmoid(layer) => layer,
                    ModuleType::Tanh(layer) => layer,
                    ModuleType::Softmax(layer) => layer,
                    ModuleType::LeakyReLU(layer) => layer,
                    ModuleType::ELU(layer) => layer,
                    ModuleType::GELU(layer) => layer,
                    ModuleType::Sequential(layer) => layer,
                    ModuleType::Conv2d(layer) => layer,
                    ModuleType::BatchNorm1d(layer) => layer,
                    ModuleType::BatchNorm2d(layer) => layer,
                    ModuleType::Dropout(layer) => layer,
                    ModuleType::Dropout2d(layer) => layer,
                };
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

    /// Estimate forward memory usage for Sequential models
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
            ModuleType::ELU(layer) => format!("ELU(alpha={})", layer.alpha()),
            ModuleType::GELU(_) => "GELU()".to_string(),
            ModuleType::Sequential(_) => "Sequential(...)".to_string(),
            ModuleType::Conv2d(layer) => format!(
                "Conv2d(in_channels={}, out_channels={}, kernel_size={:?})",
                layer.in_channels(),
                layer.out_channels(),
                layer.kernel_size()
            ),
            ModuleType::BatchNorm1d(layer) => {
                format!("BatchNorm1d(num_features={})", layer.num_features())
            }
            ModuleType::BatchNorm2d(layer) => {
                format!("BatchNorm2d(num_features={})", layer.num_features())
            }
            ModuleType::Dropout(layer) => format!("Dropout(p={})", layer.p()),
            ModuleType::Dropout2d(layer) => format!("Dropout2d(p={})", layer.p()),
        }
    }

    /// Save module state to a file (basic implementation)
    fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        // Build a SerializedModel with metadata and engine state_dict
        use engine::nn::Module as _;
        let state = match &self.inner {
            ModuleType::DenseLayer(layer) => layer.state_dict(),
            ModuleType::ReLU(layer) => layer.state_dict(),
            ModuleType::Sigmoid(layer) => layer.state_dict(),
            ModuleType::Tanh(layer) => layer.state_dict(),
            ModuleType::Softmax(layer) => layer.state_dict(),
            ModuleType::LeakyReLU(layer) => layer.state_dict(),
            ModuleType::ELU(layer) => layer.state_dict(),
            ModuleType::GELU(layer) => layer.state_dict(),
            ModuleType::Sequential(layer) => layer.state_dict(),
            ModuleType::Conv2d(layer) => layer.state_dict(),
            ModuleType::BatchNorm1d(layer) => layer.state_dict(),
            ModuleType::BatchNorm2d(layer) => layer.state_dict(),
            ModuleType::Dropout(layer) => layer.state_dict(),
            ModuleType::Dropout2d(layer) => layer.state_dict(),
        };

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
        use engine::nn::Module as _;
        let state = match &self.inner {
            ModuleType::DenseLayer(layer) => layer.state_dict(),
            ModuleType::ReLU(layer) => layer.state_dict(),
            ModuleType::Sigmoid(layer) => layer.state_dict(),
            ModuleType::Tanh(layer) => layer.state_dict(),
            ModuleType::Softmax(layer) => layer.state_dict(),
            ModuleType::LeakyReLU(layer) => layer.state_dict(),
            ModuleType::ELU(layer) => layer.state_dict(),
            ModuleType::GELU(layer) => layer.state_dict(),
            ModuleType::Sequential(layer) => layer.state_dict(),
            ModuleType::Conv2d(layer) => layer.state_dict(),
            ModuleType::BatchNorm1d(layer) => layer.state_dict(),
            ModuleType::BatchNorm2d(layer) => layer.state_dict(),
            ModuleType::Dropout(layer) => layer.state_dict(),
            ModuleType::Dropout2d(layer) => layer.state_dict(),
        };
        crate::serialization::PyStateDict::from_engine(state)
    }

    /// Load a provided StateDict into this module
    fn load_state_dict(&mut self, state: &PyStateDict, device: Option<&PyDevice>) -> PyResult<()> {
        use engine::nn::Module as _;
        let dev = device.map(|d| d.device());
        let sd_ref = crate::serialization::PyStateDict::inner_ref(state);
        let res = match &mut self.inner {
            ModuleType::DenseLayer(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::ReLU(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Sigmoid(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Tanh(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Softmax(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::LeakyReLU(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::ELU(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::GELU(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Sequential(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Conv2d(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::BatchNorm1d(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::BatchNorm2d(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Dropout(layer) => layer.load_state_dict(sd_ref, dev),
            ModuleType::Dropout2d(layer) => layer.load_state_dict(sd_ref, dev),
        };
        res.map_err(_convert_error)
    }
}

impl PyModule {
    pub fn from_dense_layer(dense_layer: DenseLayer) -> Self {
        Self {
            inner: ModuleType::DenseLayer(dense_layer),
        }
    }

    pub fn from_relu(relu: ReLU) -> Self {
        Self {
            inner: ModuleType::ReLU(relu),
        }
    }

    pub fn from_sigmoid(sigmoid: Sigmoid) -> Self {
        Self {
            inner: ModuleType::Sigmoid(sigmoid),
        }
    }

    pub fn from_tanh(tanh: Tanh) -> Self {
        Self {
            inner: ModuleType::Tanh(tanh),
        }
    }

    pub fn from_softmax(softmax: Softmax) -> Self {
        Self {
            inner: ModuleType::Softmax(softmax),
        }
    }

    pub fn from_leaky_relu(leaky_relu: LeakyReLU) -> Self {
        Self {
            inner: ModuleType::LeakyReLU(leaky_relu),
        }
    }

    pub fn from_elu(elu: ELU) -> Self {
        Self {
            inner: ModuleType::ELU(elu),
        }
    }

    pub fn from_gelu(gelu: GELU) -> Self {
        Self {
            inner: ModuleType::GELU(gelu),
        }
    }

    pub fn from_sequential(sequential: Sequential) -> Self {
        Self {
            inner: ModuleType::Sequential(sequential),
        }
    }

    pub fn from_conv2d(conv2d: Conv2d) -> Self {
        Self {
            inner: ModuleType::Conv2d(conv2d),
        }
    }

    pub fn from_batch_norm1d(batch_norm1d: BatchNorm1d) -> Self {
        Self {
            inner: ModuleType::BatchNorm1d(batch_norm1d),
        }
    }

    pub fn from_batch_norm2d(batch_norm2d: BatchNorm2d) -> Self {
        Self {
            inner: ModuleType::BatchNorm2d(batch_norm2d),
        }
    }

    pub fn from_dropout(dropout: Dropout) -> Self {
        Self {
            inner: ModuleType::Dropout(dropout),
        }
    }

    pub fn from_dropout2d(dropout: Dropout2d) -> Self {
        Self {
            inner: ModuleType::Dropout2d(dropout),
        }
    }

    pub fn to_layer(&self) -> Box<dyn Layer> {
        match &self.inner {
            ModuleType::DenseLayer(layer) => Box::new(layer.clone()),
            ModuleType::ReLU(layer) => Box::new(layer.clone()),
            ModuleType::Sigmoid(layer) => Box::new(layer.clone()),
            ModuleType::Tanh(layer) => Box::new(layer.clone()),
            ModuleType::Softmax(layer) => Box::new(layer.clone()),
            ModuleType::LeakyReLU(layer) => Box::new(layer.clone()),
            ModuleType::ELU(layer) => Box::new(layer.clone()),
            ModuleType::GELU(layer) => Box::new(layer.clone()),
            ModuleType::Sequential(_) => panic!("Nested Sequential modules are not supported"),
            ModuleType::Conv2d(layer) => Box::new(layer.clone()),
            ModuleType::BatchNorm1d(layer) => Box::new(layer.clone()),
            ModuleType::BatchNorm2d(layer) => Box::new(layer.clone()),
            ModuleType::Dropout(layer) => Box::new(layer.clone()),
            ModuleType::Dropout2d(layer) => Box::new(layer.clone()),
        }
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
    ) -> PyResult<(Self, PyModule)> {
        let bias = bias.unwrap_or(true);
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let dense_layer = DenseLayer::new(in_features, out_features, bias, device, dtype)
            .map_err(_convert_error)?;

        Ok((Self, PyModule::from_dense_layer(dense_layer)))
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

#[pymethods]
impl PyReLU {
    /// Create a new ReLU layer
    #[new]
    fn new() -> (Self, PyModule) {
        let relu = ReLU::new();
        (Self, PyModule::from_relu(relu))
    }
}

/// Sigmoid activation layer
#[pyclass(name = "Sigmoid", extends = PyModule)]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    /// Create a new Sigmoid layer
    #[new]
    fn new() -> (Self, PyModule) {
        let sigmoid = Sigmoid::new();
        (Self, PyModule::from_sigmoid(sigmoid))
    }
}

/// Tanh activation layer
#[pyclass(name = "Tanh", extends = PyModule)]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    /// Create a new Tanh layer
    #[new]
    fn new() -> (Self, PyModule) {
        let tanh = Tanh::new();
        (Self, PyModule::from_tanh(tanh))
    }
}

/// Softmax activation layer
#[pyclass(name = "Softmax", extends = PyModule)]
pub struct PySoftmax;

#[pymethods]
impl PySoftmax {
    /// Create a new Softmax layer
    #[new]
    #[pyo3(signature = (dim=None))]
    fn new(dim: Option<usize>) -> (Self, PyModule) {
        let softmax = Softmax::new(dim);
        (Self, PyModule::from_softmax(softmax))
    }

    /// Get the dimension along which softmax is computed
    #[getter]
    fn dim(slf: PyRef<Self>) -> PyResult<Option<usize>> {
        let module = slf.as_ref();
        if let ModuleType::Softmax(layer) = &module.inner {
            Ok(layer.dim())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// LeakyReLU activation layer
#[pyclass(name = "LeakyReLU", extends = PyModule)]
pub struct PyLeakyReLU;

#[pymethods]
impl PyLeakyReLU {
    /// Create a new LeakyReLU layer
    #[new]
    #[pyo3(signature = (negative_slope=None))]
    fn new(negative_slope: Option<f64>) -> (Self, PyModule) {
        let negative_slope = negative_slope.unwrap_or(0.01);
        let leaky_relu = LeakyReLU::new(Some(negative_slope));
        (Self, PyModule::from_leaky_relu(leaky_relu))
    }

    /// Get the negative slope parameter
    #[getter]
    fn negative_slope(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::LeakyReLU(layer) = &module.inner {
            Ok(layer.negative_slope())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// ELU activation layer
#[pyclass(name = "ELU", extends = PyModule)]
pub struct PyELU;

#[pymethods]
impl PyELU {
    /// Create a new ELU layer
    #[new]
    #[pyo3(signature = (alpha=None))]
    fn new(alpha: Option<f64>) -> (Self, PyModule) {
        let alpha = alpha.unwrap_or(1.0);
        let elu = ELU::new(Some(alpha));
        (Self, PyModule::from_elu(elu))
    }

    /// Get the alpha parameter
    #[getter]
    fn alpha(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::ELU(layer) = &module.inner {
            Ok(layer.alpha())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// GELU activation layer
#[pyclass(name = "GELU", extends = PyModule)]
pub struct PyGELU;

#[pymethods]
impl PyGELU {
    /// Create a new GELU layer
    #[new]
    fn new() -> (Self, PyModule) {
        let gelu = GELU::new();
        (Self, PyModule::from_gelu(gelu))
    }
}

/// Dropout layer
#[pyclass(name = "Dropout", extends = PyModule)]
pub struct PyDropout;

#[pymethods]
impl PyDropout {
    /// Create a new Dropout layer
    #[new]
    #[pyo3(signature = (p=None))]
    fn new(p: Option<f64>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let dropout = Dropout::new(Some(p)).map_err(_convert_error)?;
        Ok((Self, PyModule::from_dropout(dropout)))
    }

    /// Get the dropout probability
    #[getter]
    fn p(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::Dropout(layer) = &module.inner {
            Ok(layer.p())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// 2D Dropout layer
#[pyclass(name = "Dropout2d", extends = PyModule)]
pub struct PyDropout2d;

#[pymethods]
impl PyDropout2d {
    /// Create a new Dropout2d layer
    #[new]
    #[pyo3(signature = (p=None))]
    fn new(p: Option<f64>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let dropout = Dropout2d::new(Some(p)).map_err(_convert_error)?;
        Ok((Self, PyModule::from_dropout2d(dropout)))
    }

    /// Get the dropout probability
    #[getter]
    fn p(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::Dropout2d(layer) = &module.inner {
            Ok(layer.p())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Conv2d layer
#[pyclass(name = "Conv2d", extends = PyModule)]
pub struct PyConv2d;

#[pymethods]
impl PyConv2d {
    /// Create a new Conv2d layer
    #[new]
    #[pyo3(signature = (
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        padding=None,
        bias=None,
        device=None,
        dtype=None
    ))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: &Bound<PyAny>,
        stride: Option<&Bound<PyAny>>,
        padding: Option<&Bound<PyAny>>,
        bias: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<(Self, PyModule)> {
        let kernel_size = parse_tuple2(kernel_size)?;
        let stride = match stride {
            Some(s) => parse_tuple2(s)?,
            None => (1, 1),
        };
        let padding = match padding {
            Some(p) => parse_tuple2(p)?,
            None => (0, 0),
        };
        let bias = bias.unwrap_or(true);
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let conv2d = Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            Some(stride),
            Some(padding),
            bias,
            device,
            dtype,
        )
        .map_err(_convert_error)?;

        Ok((Self, PyModule::from_conv2d(conv2d)))
    }

    /// Get input channels count
    #[getter]
    fn in_channels(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::Conv2d(layer) = &module.inner {
            Ok(layer.in_channels())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Get output channels count
    #[getter]
    fn out_channels(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::Conv2d(layer) = &module.inner {
            Ok(layer.out_channels())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Get kernel size
    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        let module = slf.as_ref();
        if let ModuleType::Conv2d(layer) = &module.inner {
            Ok(layer.kernel_size())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// BatchNorm1d layer
#[pyclass(name = "BatchNorm1d", extends = PyModule)]
pub struct PyBatchNorm1d;

#[pymethods]
impl PyBatchNorm1d {
    /// Create a new BatchNorm1d layer
    #[new]
    #[pyo3(signature = (num_features, eps=None, momentum=None, affine=None, device=None, dtype=None))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<(Self, PyModule)> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let _affine = affine.unwrap_or(true);
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let batch_norm = BatchNorm1d::new(num_features, Some(eps), Some(momentum), device, dtype)
            .map_err(_convert_error)?;

        Ok((Self, PyModule::from_batch_norm1d(batch_norm)))
    }

    /// Get number of features
    #[getter]
    fn num_features(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::BatchNorm1d(layer) = &module.inner {
            Ok(layer.num_features())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// BatchNorm2d layer
#[pyclass(name = "BatchNorm2d", extends = PyModule)]
pub struct PyBatchNorm2d;

#[pymethods]
impl PyBatchNorm2d {
    /// Create a new BatchNorm2d layer
    #[new]
    #[pyo3(signature = (num_features, eps=None, momentum=None, affine=None, device=None, dtype=None))]
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<(Self, PyModule)> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let _affine = affine.unwrap_or(true);
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let batch_norm = BatchNorm2d::new(num_features, Some(eps), Some(momentum), device, dtype)
            .map_err(_convert_error)?;

        Ok((Self, PyModule::from_batch_norm2d(batch_norm)))
    }

    /// Get number of features
    #[getter]
    fn num_features(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::BatchNorm2d(layer) = &module.inner {
            Ok(layer.num_features())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Sequential container for layers
#[pyclass(name = "Sequential", extends = PyModule)]
pub struct PySequential;

#[pymethods]
impl PySequential {
    /// Create a new Sequential container
    #[new]
    #[pyo3(signature = (layers=None))]
    fn new(layers: Option<Vec<PyRef<PyModule>>>) -> PyResult<(Self, PyModule)> {
        let mut sequential = Sequential::new();
        if let Some(layers) = layers {
            for layer in layers {
                sequential.add_layer(layer.to_layer());
            }
        }
        Ok((Self, PyModule::from_sequential(sequential)))
    }

    /// Add a layer to the sequential container
    fn add_module(mut slf: PyRefMut<Self>, _name: &str, module: PyRef<PyModule>) -> PyResult<()> {
        let base = slf.as_mut();
        if let ModuleType::Sequential(seq) = &mut base.inner {
            seq.add_layer(module.to_layer());
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Helper function to parse data type string
fn parse_tuple2(obj: &Bound<PyAny>) -> PyResult<(usize, usize)> {
    if let Ok(val) = obj.extract::<usize>() {
        Ok((val, val))
    } else {
        obj.extract::<(usize, usize)>()
    }
}

/// MSE Loss function
#[pyclass(name = "MSELoss")]
pub struct PyMSELoss {
    inner: MSELoss,
}

#[pymethods]
impl PyMSELoss {
    /// Create a new MSE loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: MSELoss::new(reduction),
        }
    }

    /// Compute the MSE loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("MSELoss(reduction='{}')", self.inner.reduction())
    }
}

/// MAE Loss function
#[pyclass(name = "MAELoss")]
pub struct PyMAELoss {
    inner: MAELoss,
}

#[pymethods]
impl PyMAELoss {
    /// Create a new MAE loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: MAELoss::new(reduction),
        }
    }

    /// Compute the MAE loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("MAELoss(reduction='{}')", self.inner.reduction())
    }
}

/// Huber Loss function
#[pyclass(name = "HuberLoss")]
pub struct PyHuberLoss {
    inner: HuberLoss,
}

#[pymethods]
impl PyHuberLoss {
    /// Create a new Huber loss
    #[new]
    #[pyo3(signature = (delta=None, reduction=None))]
    fn new(delta: Option<f64>, reduction: Option<&str>) -> Self {
        let delta = delta.unwrap_or(1.0);
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: HuberLoss::new(delta, reduction),
        }
    }

    /// Compute the Huber loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the delta parameter
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "HuberLoss(delta={}, reduction='{}')",
            self.inner.delta(),
            self.inner.reduction()
        )
    }
}

/// Smooth L1 Loss function
#[pyclass(name = "SmoothL1Loss")]
pub struct PySmoothL1Loss {
    inner: SmoothL1Loss,
}

#[pymethods]
impl PySmoothL1Loss {
    /// Create a new Smooth L1 loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: SmoothL1Loss::new(reduction),
        }
    }

    /// Compute the Smooth L1 loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("SmoothL1Loss(reduction='{}')", self.inner.reduction())
    }
}

/// Log-cosh Loss function
#[pyclass(name = "LogCoshLoss")]
pub struct PyLogCoshLoss {
    inner: LogCoshLoss,
}

#[pymethods]
impl PyLogCoshLoss {
    /// Create a new Log-cosh loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: LogCoshLoss::new(reduction),
        }
    }

    /// Compute the Log-cosh loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("LogCoshLoss(reduction='{}')", self.inner.reduction())
    }
}

/// Cross Entropy Loss function
#[pyclass(name = "CrossEntropyLoss")]
pub struct PyCrossEntropyLoss {
    inner: CrossEntropyLoss,
}

#[pymethods]
impl PyCrossEntropyLoss {
    /// Create a new Cross Entropy loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: CrossEntropyLoss::new(reduction),
        }
    }

    /// Compute the Cross Entropy loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("CrossEntropyLoss(reduction='{}')", self.inner.reduction())
    }
}

/// Binary Cross Entropy Loss function
#[pyclass(name = "BCELoss")]
pub struct PyBCELoss {
    inner: BCELoss,
}

#[pymethods]
impl PyBCELoss {
    /// Create a new BCE loss
    #[new]
    #[pyo3(signature = (reduction=None))]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: BCELoss::new(reduction),
        }
    }

    /// Compute the BCE loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("BCELoss(reduction='{}')", self.inner.reduction())
    }
}

/// Focal Loss function
#[pyclass(name = "FocalLoss")]
pub struct PyFocalLoss {
    inner: FocalLoss,
}

#[pymethods]
impl PyFocalLoss {
    /// Create a new Focal loss
    #[new]
    #[pyo3(signature = (alpha=None, gamma=None, reduction=None))]
    fn new(alpha: Option<f64>, gamma: Option<f64>, reduction: Option<&str>) -> Self {
        let alpha = alpha.unwrap_or(0.25);
        let gamma = gamma.unwrap_or(2.0);
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: FocalLoss::new(alpha, gamma, reduction),
        }
    }

    /// Compute the Focal loss
    fn forward(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let predictions = borrow_tensor(predictions)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, predictions: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(predictions, targets)
    }

    /// Get the alpha parameter
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Get the gamma parameter
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "FocalLoss(alpha={}, gamma={}, reduction='{}')",
            self.inner.alpha(),
            self.inner.gamma(),
            self.inner.reduction()
        )
    }
}

/// Register neural network module with Python
pub fn register_nn_module(py: Python, parent_module: &Bound<Pyo3Module>) -> PyResult<()> {
    let nn_module = Pyo3Module::new(py, "nn")?;

    // Add layer classes
    nn_module.add_class::<PyModule>()?;
    nn_module.add_class::<PyDenseLayer>()?;
    nn_module.add_class::<PyReLU>()?;
    nn_module.add_class::<PySigmoid>()?;
    nn_module.add_class::<PyTanh>()?;
    nn_module.add_class::<PySoftmax>()?;
    nn_module.add_class::<PyLeakyReLU>()?;
    nn_module.add_class::<PyELU>()?;
    nn_module.add_class::<PyGELU>()?;
    nn_module.add_class::<PyDropout>()?;
    nn_module.add_class::<PyDropout2d>()?;
    nn_module.add_class::<PyConv2d>()?;
    nn_module.add_class::<PyBatchNorm1d>()?;
    nn_module.add_class::<PyBatchNorm2d>()?;
    nn_module.add_class::<PySequential>()?;

    // Add functional APIs
    nn_module.add_function(wrap_pyfunction!(dense_layer, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(conv2d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(batch_norm, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(cross_entropy, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(dropout_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(dropout2d_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(mse_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(smooth_l1_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(log_cosh_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(
        binary_cross_entropy_functional,
        &nn_module
    )?)?;

    // Add loss function classes
    nn_module.add_class::<PyMSELoss>()?;
    nn_module.add_class::<PyMAELoss>()?;
    nn_module.add_class::<PyHuberLoss>()?;
    nn_module.add_class::<PySmoothL1Loss>()?;
    nn_module.add_class::<PyLogCoshLoss>()?;
    nn_module.add_class::<PyCrossEntropyLoss>()?;
    nn_module.add_class::<PyBCELoss>()?;
    nn_module.add_class::<PyFocalLoss>()?;

    parent_module.add_submodule(&nn_module)?;
    Ok(())
}
