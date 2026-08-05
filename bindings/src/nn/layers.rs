// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyReLU {
    /// Create a new ReLU layer
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let relu = ReLU::new();
        PyClassInitializer::from(PyModule::from_relu(relu)).add_subclass(Self)
    }
}

/// Sigmoid activation layer
#[pyclass(name = "Sigmoid", extends = PyModule)]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    /// Create a new Sigmoid layer
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let sigmoid = Sigmoid::new();
        PyClassInitializer::from(PyModule::from_sigmoid(sigmoid)).add_subclass(Self)
    }
}

/// Tanh activation layer
#[pyclass(name = "Tanh", extends = PyModule)]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    /// Create a new Tanh layer
    #[new]
    fn new() -> PyClassInitializer<Self> {
        let tanh = Tanh::new();
        PyClassInitializer::from(PyModule::from_tanh(tanh)).add_subclass(Self)
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
    fn new(dim: Option<isize>) -> PyClassInitializer<Self> {
        let softmax = Softmax::new(dim);
        PyClassInitializer::from(PyModule::from_softmax(softmax)).add_subclass(Self)
    }

    /// Get the dimension along which softmax is computed
    #[getter]
    fn dim(slf: PyRef<Self>) -> PyResult<Option<isize>> {
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
    fn new(negative_slope: Option<f64>) -> PyClassInitializer<Self> {
        let negative_slope = negative_slope.unwrap_or(0.01);
        let leaky_relu = LeakyReLU::new(Some(negative_slope));
        PyClassInitializer::from(PyModule::from_leaky_relu(leaky_relu)).add_subclass(Self)
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
    fn new(alpha: Option<f64>) -> PyClassInitializer<Self> {
        let alpha = alpha.unwrap_or(1.0);
        let elu = ELU::new(Some(alpha));
        PyClassInitializer::from(PyModule::from_elu(elu)).add_subclass(Self)
    }

    /// Get the alpha parameter
    #[getter]
    fn alpha(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::Elu(layer) = &module.inner {
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
    fn new() -> PyClassInitializer<Self> {
        let gelu = GELU::new();
        PyClassInitializer::from(PyModule::from_gelu(gelu)).add_subclass(Self)
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
    fn new(p: Option<f64>) -> PyResult<PyClassInitializer<Self>> {
        let p = p.unwrap_or(0.5);
        let dropout = Dropout::new(Some(p)).map_err(_convert_error)?;
        Ok(PyClassInitializer::from(PyModule::from_dropout(dropout)).add_subclass(Self))
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
    fn new(p: Option<f64>) -> PyResult<PyClassInitializer<Self>> {
        let p = p.unwrap_or(0.5);
        let dropout = Dropout2d::new(Some(p)).map_err(_convert_error)?;
        Ok(PyClassInitializer::from(PyModule::from_dropout2d(dropout)).add_subclass(Self))
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

/// 1-D convolution over the last dimension, learning `out_channels` filters.
#[pyclass(name = "Conv1d", extends = PyModule)]
pub struct PyConv1d;

#[pymethods]
impl PyConv1d {
    /// Create a new Conv1d layer
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, bias=true, device=None, dtype=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let layer = Conv1d::new(
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
        Ok(PyClassInitializer::from(PyModule::from_conv1d(layer)).add_subclass(Self))
    }

    #[getter]
    fn in_channels(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::Conv1d(layer) => Ok(layer.in_channels()),
            _ => Err(PyTypeError::new_err("Not a Conv1d layer")),
        }
    }

    #[getter]
    fn out_channels(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::Conv1d(layer) => Ok(layer.out_channels()),
            _ => Err(PyTypeError::new_err("Not a Conv1d layer")),
        }
    }

    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::Conv1d(layer) => Ok(layer.kernel_size()),
            _ => Err(PyTypeError::new_err("Not a Conv1d layer")),
        }
    }

    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::Conv1d(layer) => Ok(layer.stride()),
            _ => Err(PyTypeError::new_err("Not a Conv1d layer")),
        }
    }

    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::Conv1d(layer) => Ok(layer.padding()),
            _ => Err(PyTypeError::new_err("Not a Conv1d layer")),
        }
    }
}

/// MaxPool1d layer
#[pyclass(name = "MaxPool1d", extends = PyModule)]
pub struct PyMaxPool1d;

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        // Pooling defaults its stride to the window, unlike convolution.
        let layer = MaxPool1d::new(kernel_size, stride, Some(padding));
        Ok(PyClassInitializer::from(PyModule::from_max_pool1d(layer)).add_subclass(Self))
    }

    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool1d(layer) => Ok(layer.kernel_size()),
            _ => Err(PyTypeError::new_err("Not a MaxPool1d layer")),
        }
    }

    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool1d(layer) => Ok(layer.stride()),
            _ => Err(PyTypeError::new_err("Not a MaxPool1d layer")),
        }
    }

    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool1d(layer) => Ok(layer.padding()),
            _ => Err(PyTypeError::new_err("Not a MaxPool1d layer")),
        }
    }
}

/// AvgPool1d layer
#[pyclass(name = "AvgPool1d", extends = PyModule)]
pub struct PyAvgPool1d;

#[pymethods]
impl PyAvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0, count_include_pad=true))]
    fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        count_include_pad: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layer = AvgPool1d::new(kernel_size, stride, Some(padding), count_include_pad);
        Ok(PyClassInitializer::from(PyModule::from_avg_pool1d(layer)).add_subclass(Self))
    }

    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool1d(layer) => Ok(layer.kernel_size()),
            _ => Err(PyTypeError::new_err("Not an AvgPool1d layer")),
        }
    }

    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool1d(layer) => Ok(layer.stride()),
            _ => Err(PyTypeError::new_err("Not an AvgPool1d layer")),
        }
    }

    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<usize> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool1d(layer) => Ok(layer.padding()),
            _ => Err(PyTypeError::new_err("Not an AvgPool1d layer")),
        }
    }

    #[getter]
    fn count_include_pad(slf: PyRef<Self>) -> PyResult<bool> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool1d(layer) => Ok(layer.count_include_pad()),
            _ => Err(PyTypeError::new_err("Not an AvgPool1d layer")),
        }
    }
}

/// MaxPool2d layer
#[pyclass(name = "MaxPool2d", extends = PyModule)]
pub struct PyMaxPool2d;

#[pymethods]
impl PyMaxPool2d {
    /// Create a new MaxPool2d layer
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: &Bound<PyAny>,
        stride: Option<&Bound<PyAny>>,
        padding: Option<&Bound<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let kernel_size = parse_tuple2(kernel_size)?;
        // Pooling defaults its stride to the window, unlike convolution.
        let stride = match stride {
            Some(s) => Some(parse_tuple2(s)?),
            None => None,
        };
        let padding = match padding {
            Some(p) => Some(parse_tuple2(p)?),
            None => None,
        };
        let layer = MaxPool2d::new(kernel_size, stride, padding);
        Ok(PyClassInitializer::from(PyModule::from_max_pool2d(layer)).add_subclass(Self))
    }

    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool2d(layer) => Ok(layer.kernel_size()),
            _ => Err(PyTypeError::new_err("Not a MaxPool2d layer")),
        }
    }

    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool2d(layer) => Ok(layer.stride()),
            _ => Err(PyTypeError::new_err("Not a MaxPool2d layer")),
        }
    }

    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        match &slf.as_ref().inner {
            ModuleType::MaxPool2d(layer) => Ok(layer.padding()),
            _ => Err(PyTypeError::new_err("Not a MaxPool2d layer")),
        }
    }
}

/// AvgPool2d layer
#[pyclass(name = "AvgPool2d", extends = PyModule)]
pub struct PyAvgPool2d;

#[pymethods]
impl PyAvgPool2d {
    /// Create a new AvgPool2d layer
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None, count_include_pad=true))]
    fn new(
        kernel_size: &Bound<PyAny>,
        stride: Option<&Bound<PyAny>>,
        padding: Option<&Bound<PyAny>>,
        count_include_pad: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let kernel_size = parse_tuple2(kernel_size)?;
        let stride = match stride {
            Some(s) => Some(parse_tuple2(s)?),
            None => None,
        };
        let padding = match padding {
            Some(p) => Some(parse_tuple2(p)?),
            None => None,
        };
        let layer = AvgPool2d::new(kernel_size, stride, padding, Some(count_include_pad));
        Ok(PyClassInitializer::from(PyModule::from_avg_pool2d(layer)).add_subclass(Self))
    }

    #[getter]
    fn kernel_size(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool2d(layer) => Ok(layer.kernel_size()),
            _ => Err(PyTypeError::new_err("Not an AvgPool2d layer")),
        }
    }

    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool2d(layer) => Ok(layer.stride()),
            _ => Err(PyTypeError::new_err("Not an AvgPool2d layer")),
        }
    }

    #[getter]
    fn count_include_pad(slf: PyRef<Self>) -> PyResult<bool> {
        match &slf.as_ref().inner {
            ModuleType::AvgPool2d(layer) => Ok(layer.count_include_pad()),
            _ => Err(PyTypeError::new_err("Not an AvgPool2d layer")),
        }
    }
}

/// 2-D convolution over the last two dimensions, learning `out_channels` filters.
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
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: &Bound<PyAny>,
        stride: Option<&Bound<PyAny>>,
        padding: Option<&Bound<PyAny>>,
        bias: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
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
        let device = resolve_device(device)?;
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

        Ok(PyClassInitializer::from(PyModule::from_conv2d(conv2d)).add_subclass(Self))
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let _affine = affine.unwrap_or(true);
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let batch_norm = BatchNorm1d::new(num_features, Some(eps), Some(momentum), device, dtype)
            .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_batch_norm1d(batch_norm)).add_subclass(Self))
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let _affine = affine.unwrap_or(true);
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let batch_norm = BatchNorm2d::new(num_features, Some(eps), Some(momentum), device, dtype)
            .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_batch_norm2d(batch_norm)).add_subclass(Self))
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

/// Embedding lookup table
#[pyclass(name = "Embedding", extends = PyModule)]
pub struct PyEmbedding;

#[pymethods]
impl PyEmbedding {
    /// Create a new Embedding layer
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None, device=None, dtype=None))]
    fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let embedding = Embedding::new(num_embeddings, embedding_dim, padding_idx, device, dtype)
            .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_embedding(embedding)).add_subclass(Self))
    }

    /// Vocabulary size
    #[getter]
    fn num_embeddings(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::Embedding(layer) = &module.inner {
            Ok(layer.num_embeddings())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Width of each embedding vector
    #[getter]
    fn embedding_dim(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::Embedding(layer) = &module.inner {
            Ok(layer.embedding_dim())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Token id held at zero, if any
    #[getter]
    fn padding_idx(slf: PyRef<Self>) -> PyResult<Option<usize>> {
        let module = slf.as_ref();
        if let ModuleType::Embedding(layer) = &module.inner {
            Ok(layer.padding_idx())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// The embedding matrix
    #[getter]
    fn weight(slf: PyRef<Self>) -> PyResult<PyTensor> {
        let module = slf.as_ref();
        if let ModuleType::Embedding(layer) = &module.inner {
            Ok(PyTensor::from_tensor(layer.weight().clone()))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Layer normalization
#[pyclass(name = "LayerNorm", extends = PyModule)]
pub struct PyLayerNorm;

#[pymethods]
impl PyLayerNorm {
    /// Create a new LayerNorm layer
    #[new]
    #[pyo3(signature = (normalized_shape, eps=None, elementwise_affine=None, device=None, dtype=None))]
    fn new(
        normalized_shape: &Bound<PyAny>,
        eps: Option<f64>,
        elementwise_affine: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let shape = parse_normalized_shape_arg(normalized_shape)?;
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let layer = LayerNorm::new(
            shape,
            eps,
            elementwise_affine.unwrap_or(true),
            device,
            dtype,
        )
        .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_layer_norm(layer)).add_subclass(Self))
    }

    /// Dimensions normalized over
    #[getter]
    fn normalized_shape(slf: PyRef<Self>) -> PyResult<Vec<usize>> {
        let module = slf.as_ref();
        if let ModuleType::LayerNorm(layer) = &module.inner {
            Ok(layer.normalized_shape().to_vec())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Numerical stability epsilon
    #[getter]
    fn eps(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::LayerNorm(layer) = &module.inner {
            Ok(layer.eps())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Root-mean-square layer normalization
#[pyclass(name = "RMSNorm", extends = PyModule)]
pub struct PyRMSNorm;

#[pymethods]
impl PyRMSNorm {
    /// Create a new RMSNorm layer
    #[new]
    #[pyo3(signature = (normalized_shape, eps=None, elementwise_affine=None, device=None, dtype=None))]
    fn new(
        normalized_shape: &Bound<PyAny>,
        eps: Option<f64>,
        elementwise_affine: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let shape = parse_normalized_shape_arg(normalized_shape)?;
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let layer = RMSNorm::new(
            shape,
            eps,
            elementwise_affine.unwrap_or(true),
            device,
            dtype,
        )
        .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_rms_norm(layer)).add_subclass(Self))
    }

    /// Dimensions normalized over
    #[getter]
    fn normalized_shape(slf: PyRef<Self>) -> PyResult<Vec<usize>> {
        let module = slf.as_ref();
        if let ModuleType::RMSNorm(layer) = &module.inner {
            Ok(layer.normalized_shape().to_vec())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Numerical stability epsilon
    #[getter]
    fn eps(slf: PyRef<Self>) -> PyResult<f64> {
        let module = slf.as_ref();
        if let ModuleType::RMSNorm(layer) = &module.inner {
            Ok(layer.eps())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Multi-head attention block
/// Shared constructor and state-carrying forward for the recurrent layers.
macro_rules! recurrent_class {
    ($py_name:literal, $ty:ident, $ctor:ident, $doc:literal, $returns_cell:literal) => {
        #[doc = $doc]
        #[pyclass(name = $py_name, extends = PyModule)]
        pub struct $ty;

        #[pymethods]
        impl $ty {
            #[new]
            // Kept on one line: rustfmt re-indents attribute bodies inside a
            // macro on every pass and never converges.
            #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false, device=None, dtype=None))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                input_size: usize,
                hidden_size: usize,
                num_layers: usize,
                bias: bool,
                batch_first: bool,
                bidirectional: bool,
                device: Option<&PyDevice>,
                dtype: Option<&str>,
            ) -> PyResult<PyClassInitializer<Self>> {
                let device = resolve_device(device)?;
                let dtype = dtype::resolve_dtype_arg(dtype)?;
                let layer = Recurrent::$ctor(
                    input_size,
                    hidden_size,
                    num_layers,
                    bias,
                    batch_first,
                    bidirectional,
                    device,
                    dtype,
                )
                .map_err(_convert_error)?;
                Ok(PyClassInitializer::from(PyModule::from_recurrent(layer)).add_subclass(Self))
            }

            /// Run the stack and return the final states alongside the output.
            ///
            /// `hx` supplies the initial hidden state shaped
            /// `(num_layers, batch, hidden_size)`; zeros are used when omitted.
            /// LSTM additionally accepts `cx` and returns `(output, (h_n, c_n))`;
            /// GRU returns `(output, h_n)`.
            #[pyo3(signature = (input, hx=None, cx=None))]
            fn forward_with_state<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
                input: &Bound<'py, PyAny>,
                hx: Option<&Bound<'py, PyAny>>,
                cx: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let module = slf.as_ref();
                let ModuleType::Recurrent(layer) = &module.inner else {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Invalid layer type",
                    ));
                };

                let x = borrow_tensor(input)?;
                let h0 = borrow_optional_tensor(hx)?;
                let c0 = borrow_optional_tensor(cx)?;
                if h0.is_none() && c0.is_some() {
                    return Err(PyValueError::new_err(
                        "cx was given without hx; pass both initial states or neither",
                    ));
                }
                let state = h0
                    .as_ref()
                    .map(|h| (h.tensor(), c0.as_ref().map(|c| c.tensor())));

                let (output, h_n, c_n) = layer
                    .forward_with_state(x.tensor(), state)
                    .map_err(_convert_error)?;

                let output = Py::new(py, PyTensor::from_tensor(output))?.into_any();
                let h_n = Py::new(py, PyTensor::from_tensor(h_n))?.into_any();
                if $returns_cell {
                    let c_n = c_n.ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            "LSTM did not produce a cell state",
                        )
                    })?;
                    let c_n = Py::new(py, PyTensor::from_tensor(c_n))?.into_any();
                    let states = PyTuple::new(py, [h_n, c_n])?.into_any().unbind();
                    Ok(PyTuple::new(py, [output, states])?.into_any().unbind())
                } else {
                    Ok(PyTuple::new(py, [output, h_n])?.into_any().unbind())
                }
            }

            /// Width of each input vector
            #[getter]
            fn input_size(slf: PyRef<Self>) -> PyResult<usize> {
                Self::with_layer(slf, |l| l.input_size())
            }

            /// Width of the hidden state
            #[getter]
            fn hidden_size(slf: PyRef<Self>) -> PyResult<usize> {
                Self::with_layer(slf, |l| l.hidden_size())
            }

            /// Number of stacked layers
            #[getter]
            fn num_layers(slf: PyRef<Self>) -> PyResult<usize> {
                Self::with_layer(slf, |l| l.num_layers())
            }

            /// Whether inputs are `(batch, seq, feature)`
            #[getter]
            fn batch_first(slf: PyRef<Self>) -> PyResult<bool> {
                Self::with_layer(slf, |l| l.batch_first())
            }

            /// Whether the layers carry additive biases
            #[getter]
            fn bias(slf: PyRef<Self>) -> PyResult<bool> {
                Self::with_layer(slf, |l| l.has_bias())
            }

            /// Whether each layer also runs over the reversed sequence
            #[getter]
            fn bidirectional(slf: PyRef<Self>) -> PyResult<bool> {
                Self::with_layer(slf, |l| l.bidirectional())
            }

            /// Width of the output feature axis (`hidden_size * directions`)
            #[getter]
            fn output_size(slf: PyRef<Self>) -> PyResult<usize> {
                Self::with_layer(slf, |l| l.output_size())
            }
        }

        impl $ty {
            fn with_layer<T>(slf: PyRef<Self>, f: impl Fn(&Recurrent) -> T) -> PyResult<T> {
                let module = slf.as_ref();
                let ModuleType::Recurrent(layer) = &module.inner else {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Invalid layer type",
                    ));
                };
                Ok(f(layer))
            }
        }
    };
}

recurrent_class!(
    "LSTM",
    PyLSTM,
    lstm,
    "Long Short-Term Memory layer. Inputs are `(seq, batch, input_size)`, or \
     `(batch, seq, input_size)` when `batch_first`.",
    true
);

recurrent_class!(
    "GRU",
    PyGRU,
    gru,
    "Gated Recurrent Unit layer. Inputs are `(seq, batch, input_size)`, or \
     `(batch, seq, input_size)` when `batch_first`.",
    false
);

/// Scaled dot-product attention over `num_heads` heads.
///
/// Called with one tensor it is self-attention; called with separate query,
/// key and value sequences it is cross-attention. `embed_dim` must divide
/// evenly by `num_heads`.
#[pyclass(name = "MultiheadAttention", extends = PyModule)]
pub struct PyMultiheadAttention;

#[pymethods]
impl PyMultiheadAttention {
    /// Create a new MultiheadAttention layer
    #[new]
    #[pyo3(signature = (embed_dim, num_heads, bias=None, is_causal=None, device=None, dtype=None))]
    fn new(
        embed_dim: usize,
        num_heads: usize,
        bias: Option<bool>,
        is_causal: Option<bool>,
        device: Option<&PyDevice>,
        dtype: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let device = resolve_device(device)?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;

        let mha = MultiheadAttention::new(
            embed_dim,
            num_heads,
            bias.unwrap_or(true),
            is_causal.unwrap_or(false),
            device,
            dtype,
        )
        .map_err(_convert_error)?;

        Ok(PyClassInitializer::from(PyModule::from_multihead_attention(mha)).add_subclass(Self))
    }

    /// Attention over separate query/key/value sequences (cross-attention).
    ///
    /// `key` and `value` must share a batch size and sequence length; `query`
    /// may have its own sequence length, and the output follows it. `attn_mask`
    /// broadcasts to the per-head scores `(batch, heads, query_seq, key_seq)`.
    #[pyo3(signature = (query, key, value, attn_mask=None, is_causal=false))]
    fn forward_qkv(
        slf: PyRef<Self>,
        query: &Bound<PyAny>,
        key: &Bound<PyAny>,
        value: &Bound<PyAny>,
        attn_mask: Option<&Bound<PyAny>>,
        is_causal: bool,
    ) -> PyResult<PyTensor> {
        let module = slf.as_ref();
        let ModuleType::MultiheadAttention(layer) = &module.inner else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ));
        };

        let q = borrow_tensor(query)?;
        let k = borrow_tensor(key)?;
        let v = borrow_tensor(value)?;
        let mask = borrow_optional_tensor(attn_mask)?;

        let result = layer
            .forward_qkv(
                q.tensor(),
                k.tensor(),
                v.tensor(),
                mask.as_deref().map(|m| m.tensor()),
                is_causal,
            )
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Model width
    #[getter]
    fn embed_dim(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::MultiheadAttention(layer) = &module.inner {
            Ok(layer.embed_dim())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Number of attention heads
    #[getter]
    fn num_heads(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::MultiheadAttention(layer) = &module.inner {
            Ok(layer.num_heads())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Width of each head
    #[getter]
    fn head_dim(slf: PyRef<Self>) -> PyResult<usize> {
        let module = slf.as_ref();
        if let ModuleType::MultiheadAttention(layer) = &module.inner {
            Ok(layer.head_dim())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }

    /// Whether forward applies an autoregressive mask
    #[getter]
    fn is_causal(slf: PyRef<Self>) -> PyResult<bool> {
        let module = slf.as_ref();
        if let ModuleType::MultiheadAttention(layer) = &module.inner {
            Ok(layer.is_causal())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ))
        }
    }
}

/// Accept either an int or a sequence of ints for `normalized_shape`.
fn parse_normalized_shape_arg(arg: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(value) = arg.extract::<usize>() {
        return Ok(vec![value]);
    }
    let seq = arg.extract::<Vec<usize>>().map_err(|_| {
        PyValueError::new_err("normalized_shape must be an int or a sequence of ints")
    })?;
    if seq.is_empty() {
        return Err(PyValueError::new_err(
            "normalized_shape must contain at least one dimension",
        ));
    }
    Ok(seq)
}

/// Sequential container for layers
#[pyclass(name = "Sequential", extends = PyModule)]
pub struct PySequential;

#[pymethods]
impl PySequential {
    /// Create a new Sequential container
    #[new]
    #[pyo3(signature = (layers=None))]
    fn new(layers: Option<Vec<PyRef<PyModule>>>) -> PyResult<PyClassInitializer<Self>> {
        let sequential = if let Some(layers) = layers {
            let mut layer_objects = Vec::with_capacity(layers.len());
            for layer in layers {
                layer_objects.push(layer.to_layer()?);
            }
            Sequential::from_layers(layer_objects)
        } else {
            Sequential::new()
        };

        Ok(PyClassInitializer::from(PyModule::from_sequential(sequential)).add_subclass(Self))
    }

    /// Add a layer to the sequential container
    fn add_module(mut slf: PyRefMut<Self>, _name: &str, module: PyRef<PyModule>) -> PyResult<()> {
        if !matches!(slf.as_ref().inner, ModuleType::Sequential(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid layer type",
            ));
        }

        let layer = module.to_layer()?;

        if let ModuleType::Sequential(seq) = &mut slf.as_mut().inner {
            seq.add_layer(layer);
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

/// Binary Cross Entropy Loss function taking logits instead of probabilities
#[pyclass(name = "BCEWithLogitsLoss")]
pub struct PyBCEWithLogitsLoss {
    inner: BCEWithLogitsLoss,
}

#[pymethods]
impl PyBCEWithLogitsLoss {
    /// Create a new BCE-with-logits loss
    #[new]
    #[pyo3(signature = (reduction=None, pos_weight=None))]
    fn new(reduction: Option<&str>, pos_weight: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let reduction = reduction.unwrap_or("mean");
        let inner = match pos_weight {
            Some(w) => {
                let w = borrow_tensor(w)?;
                BCEWithLogitsLoss::with_pos_weight(reduction, w.tensor().clone())
            }
            None => BCEWithLogitsLoss::new(reduction),
        };
        Ok(Self { inner })
    }

    /// Compute the loss from raw logits
    fn forward(&self, logits: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        let logits = borrow_tensor(logits)?;
        let targets = borrow_tensor(targets)?;
        let result = self
            .inner
            .forward(logits.tensor(), targets.tensor())
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    #[pyo3(name = "__call__")]
    fn call(&self, logits: &Bound<PyAny>, targets: &Bound<PyAny>) -> PyResult<PyTensor> {
        self.forward(logits, targets)
    }

    /// Get the reduction mode
    #[getter]
    fn reduction(&self) -> &str {
        self.inner.reduction()
    }

    /// Get the positive-class weight, if one was set
    #[getter]
    fn pos_weight(&self) -> Option<PyTensor> {
        self.inner
            .pos_weight()
            .map(|w: &engine::tensor::Tensor| PyTensor::from_tensor(w.clone()))
    }

    /// String representation
    fn __repr__(&self) -> String {
        match self.inner.pos_weight() {
            Some(_) => format!(
                "BCEWithLogitsLoss(reduction='{}', pos_weight=...)",
                self.inner.reduction()
            ),
            None => format!("BCEWithLogitsLoss(reduction='{}')", self.inner.reduction()),
        }
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
    nn_module.setattr(
        "__doc__",
        "Neural network layers, losses and the functional forms of both.",
    )?;

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
    nn_module.add_class::<PyMaxPool2d>()?;
    nn_module.add_class::<PyAvgPool2d>()?;
    nn_module.add_class::<PyBatchNorm1d>()?;
    nn_module.add_class::<PyBatchNorm2d>()?;
    nn_module.add_class::<PyEmbedding>()?;
    nn_module.add_class::<PyLayerNorm>()?;
    nn_module.add_class::<PyRMSNorm>()?;
    nn_module.add_class::<PyMultiheadAttention>()?;
    nn_module.add_class::<PySequential>()?;

    // Add functional APIs
    nn_module.add_function(wrap_pyfunction!(dense_layer, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(conv2d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(conv1d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(max_pool1d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(avg_pool1d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(max_pool2d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(avg_pool2d, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(batch_norm, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(cross_entropy, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(dropout_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(dropout2d_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(mse_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(smooth_l1_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(huber_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(l1_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(kl_div_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(focal_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(log_cosh_loss_functional, &nn_module)?)?;
    nn_module.add_function(wrap_pyfunction!(
        binary_cross_entropy_functional,
        &nn_module
    )?)?;
    nn_module.add_function(wrap_pyfunction!(
        binary_cross_entropy_with_logits_functional,
        &nn_module
    )?)?;

    // Add loss function classes
    nn_module.add_class::<PyMSELoss>()?;
    nn_module.add_class::<PyMAELoss>()?;
    nn_module.add_class::<PyHuberLoss>()?;
    nn_module.add_class::<PySmoothL1Loss>()?;
    nn_module.add_class::<PyLogCoshLoss>()?;
    nn_module.add_class::<PyCrossEntropyLoss>()?;
    nn_module.add_class::<PyConv1d>()?;
    nn_module.add_class::<PyMaxPool1d>()?;
    nn_module.add_class::<PyAvgPool1d>()?;
    nn_module.add_class::<PyLSTM>()?;
    nn_module.add_class::<PyGRU>()?;
    nn_module.add_class::<PyBCELoss>()?;
    nn_module.add_class::<PyBCEWithLogitsLoss>()?;
    nn_module.add_class::<PyFocalLoss>()?;

    // Gradient clipping lives here because that is where PyTorch users look
    // (`torch.nn.utils.clip_grad_norm_`), and because `nn` is already this
    // library's home for free functions.
    crate::grad_utils::register(&nn_module)?;
    super::init::register(py, &nn_module)?;

    parent_module.add_submodule(&nn_module)?;
    Ok(())
}
