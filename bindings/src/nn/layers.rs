// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
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
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
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
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
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
        let sequential = if let Some(layers) = layers {
            let mut layer_objects = Vec::with_capacity(layers.len());
            for layer in layers {
                layer_objects.push(layer.to_layer()?);
            }
            Sequential::from_layers(layer_objects)
        } else {
            Sequential::new()
        };

        Ok((Self, PyModule::from_sequential(sequential)))
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
