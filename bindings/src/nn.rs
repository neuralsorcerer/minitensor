// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::error::_convert_error;
use crate::serialization::PyStateDict;
use crate::tensor::PyTensor;
use engine::nn::loss::{BCELoss, CrossEntropyLoss, FocalLoss};
use engine::nn::{
    activation::{LeakyReLU, ELU, GELU},
    conv::Conv2d,
    dropout::Dropout,
    normalization::{BatchNorm1d, BatchNorm2d},
    DenseLayer, HuberLoss, Layer, MAELoss, MSELoss, ReLU, Sequential, Sigmoid, Softmax, Tanh,
};
use engine::operations::batch_norm as batch_norm_op;
use engine::operations::conv2d as conv2d_op;
use engine::operations::loss::cross_entropy as cross_entropy_op;
use engine::serialization::{ModelMetadata, ModelSerializer, SerializationFormat, SerializedModel};
use engine::tensor::DataType;
use engine::Device;
use pyo3::prelude::*;
use pyo3::exceptions::PyIndexError;
use pyo3::types::{PyAny, PyModule as Pyo3Module};

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
fn conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let stride = stride.unwrap_or((1, 1));
    let padding = padding.unwrap_or((0, 0));
    let bias_tensor = bias.map(|b| b.tensor());
    let result = conv2d_op(
        input.tensor(),
        weight.tensor(),
        bias_tensor,
        stride,
        padding,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=true, momentum=0.1, eps=1e-5))]
fn batch_norm(
    input: &PyTensor,
    running_mean: Option<PyRefMut<PyTensor>>,
    running_var: Option<PyRefMut<PyTensor>>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> PyResult<PyTensor> {
    let mut rm_opt = running_mean;
    let mut rv_opt = running_var;
    let rm_tensor = rm_opt.as_mut().map(|t| t.tensor_mut());
    let rv_tensor = rv_opt.as_mut().map(|t| t.tensor_mut());
    let result = batch_norm_op(
        input.tensor(),
        rm_tensor,
        rv_tensor,
        weight.map(|w| w.tensor()),
        bias.map(|b| b.tensor()),
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
    input: &PyTensor,
    target: &PyTensor,
    reduction: &str,
    dim: isize,
) -> PyResult<PyTensor> {
    let ndim = input.tensor().ndim() as isize;
    let axis = if dim < 0 { ndim + dim } else { dim };
    if axis < 0 || axis as usize >= ndim as usize {
        return Err(PyIndexError::new_err("dim out of range"));
    }
    let result = cross_entropy_op(
        input.tensor(),
        target.tensor(),
        reduction,
        axis as usize,
    )
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
}

#[pymethods]
impl PyModule {
    /// Forward pass through the module
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = match &mut self.inner {
            ModuleType::DenseLayer(layer) => layer.forward(input.tensor()),
            ModuleType::ReLU(layer) => layer.forward(input.tensor()),
            ModuleType::Sigmoid(layer) => layer.forward(input.tensor()),
            ModuleType::Tanh(layer) => layer.forward(input.tensor()),
            ModuleType::Softmax(layer) => layer.forward(input.tensor()),
            ModuleType::LeakyReLU(layer) => layer.forward(input.tensor()),
            ModuleType::ELU(layer) => layer.forward(input.tensor()),
            ModuleType::GELU(layer) => layer.forward(input.tensor()),
            ModuleType::Sequential(layer) => layer.forward(input.tensor()),
            ModuleType::Conv2d(layer) => layer.forward(input.tensor()),
            ModuleType::BatchNorm1d(layer) => layer.forward(input.tensor()),
            ModuleType::BatchNorm2d(layer) => layer.forward(input.tensor()),
            ModuleType::Dropout(layer) => layer.forward(input.tensor()),
        }
        .map_err(_convert_error)?;

        Ok(PyTensor::from_tensor(result))
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
}

/// DenseLayer (fully connected) layer
#[pyclass(name = "DenseLayer", extends = PyModule)]
pub struct PyDenseLayer;

#[pymethods]
impl PyDenseLayer {
    /// Create a new dense layer
    #[new]
    fn new(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<(Self, PyModule)> {
        let bias = bias.unwrap_or(true);
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;

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

/// Conv2d layer
#[pyclass(name = "Conv2d", extends = PyModule)]
pub struct PyConv2d;

#[pymethods]
impl PyConv2d {
    /// Create a new Conv2d layer
    #[new]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: &PyAny,
        stride: Option<&PyAny>,
        padding: Option<&PyAny>,
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
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;

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
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;

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
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;

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
    fn new(_layers: Option<Vec<PyRef<PyModule>>>) -> PyResult<(Self, PyModule)> {
        // For now, create an empty sequential
        // In a full implementation, we'd convert the Python modules to Rust layers
        let sequential = Sequential::new();
        Ok((Self, PyModule::from_sequential(sequential)))
    }

    /// Add a layer to the sequential container
    fn add_module(&mut self, _name: &str, _module: PyRef<PyModule>) -> PyResult<()> {
        // This would require a more complex implementation to handle dynamic layer addition
        // For now, we'll return a not implemented error
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Dynamic layer addition not yet implemented",
        ))
    }
}

/// Helper function to parse data type string
fn parse_tuple2(obj: &PyAny) -> PyResult<(usize, usize)> {
    if let Ok(val) = obj.extract::<usize>() {
        Ok((val, val))
    } else {
        obj.extract::<(usize, usize)>()
    }
}

fn parse_dtype(dtype_str: &str) -> PyResult<DataType> {
    match dtype_str {
        "float32" | "f32" => Ok(DataType::Float32),
        "float64" | "f64" => Ok(DataType::Float64),
        "int32" | "i32" => Ok(DataType::Int32),
        "int64" | "i64" => Ok(DataType::Int64),
        "bool" => Ok(DataType::Bool),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported data type: {}",
            dtype_str
        ))),
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
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: MSELoss::new(reduction),
        }
    }

    /// Compute the MSE loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: MAELoss::new(reduction),
        }
    }

    /// Compute the MAE loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
    fn new(delta: Option<f64>, reduction: Option<&str>) -> Self {
        let delta = delta.unwrap_or(1.0);
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: HuberLoss::new(delta, reduction),
        }
    }

    /// Compute the Huber loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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

/// Cross Entropy Loss function
#[pyclass(name = "CrossEntropyLoss")]
pub struct PyCrossEntropyLoss {
    inner: CrossEntropyLoss,
}

#[pymethods]
impl PyCrossEntropyLoss {
    /// Create a new Cross Entropy loss
    #[new]
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: CrossEntropyLoss::new(reduction),
        }
    }

    /// Compute the Cross Entropy loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
    fn new(reduction: Option<&str>) -> Self {
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: BCELoss::new(reduction),
        }
    }

    /// Compute the BCE loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
    fn new(alpha: Option<f64>, gamma: Option<f64>, reduction: Option<&str>) -> Self {
        let alpha = alpha.unwrap_or(0.25);
        let gamma = gamma.unwrap_or(2.0);
        let reduction = reduction.unwrap_or("mean");
        Self {
            inner: FocalLoss::new(alpha, gamma, reduction),
        }
    }

    /// Compute the Focal loss
    fn forward(&self, predictions: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .forward(predictions.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
pub fn register_nn_module(py: Python, parent_module: &Pyo3Module) -> PyResult<()> {
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
    nn_module.add_class::<PyConv2d>()?;
    nn_module.add_class::<PyBatchNorm1d>()?;
    nn_module.add_class::<PyBatchNorm2d>()?;
    nn_module.add_class::<PySequential>()?;

    // Add functional APIs
    nn_module.add_function(wrap_pyfunction!(conv2d, nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(batch_norm, nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(cross_entropy, nn_module)?)?;

    // Add loss function classes
    nn_module.add_class::<PyMSELoss>()?;
    nn_module.add_class::<PyMAELoss>()?;
    nn_module.add_class::<PyHuberLoss>()?;
    nn_module.add_class::<PyCrossEntropyLoss>()?;
    nn_module.add_class::<PyBCELoss>()?;
    nn_module.add_class::<PyFocalLoss>()?;

    parent_module.add_submodule(nn_module)?;
    Ok(())
}
