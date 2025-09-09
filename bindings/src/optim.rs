// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::optim::{Adam, Optimizer, RMSprop, SGD};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule as Pyo3Module};

/// Base class for optimizers
#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    inner: OptimizerType,
}

enum OptimizerType {
    SGD(SGD),
    Adam(Adam),
    RMSprop(RMSprop),
}

#[pymethods]
impl PyOptimizer {
    /// Perform a single optimization step on the provided parameters.
    fn step(&mut self, parameters: &Bound<PyList>) -> PyResult<()> {
        // Borrow all tensors mutably so we can pass raw tensor references to the
        // underlying Rust optimizers. We first collect the PyRefMut guards to
        // ensure the lifetimes of the borrows outlive the optimizer call.
        let mut borrowed: Vec<PyRefMut<PyTensor>> = Vec::with_capacity(parameters.len());
        for obj in parameters.iter() {
            borrowed.push(obj.extract::<PyRefMut<PyTensor>>()?);
        }

        // Extract mutable references to the inner Tensor objects.
        let mut tensor_refs: Vec<&mut engine::tensor::Tensor> =
            borrowed.iter_mut().map(|t| t.tensor_mut()).collect();

        // Dispatch to the correct optimizer implementation.
        let result = match &mut self.inner {
            OptimizerType::SGD(opt) => opt.step(tensor_refs.as_mut_slice()),
            OptimizerType::Adam(opt) => opt.step(tensor_refs.as_mut_slice()),
            OptimizerType::RMSprop(opt) => opt.step(tensor_refs.as_mut_slice()),
        };

        result.map_err(_convert_error)
    }

    /// Zero out gradients for the provided parameters.
    fn zero_grad(&self, parameters: &Bound<PyList>) -> PyResult<()> {
        let mut borrowed: Vec<PyRefMut<PyTensor>> = Vec::with_capacity(parameters.len());
        for obj in parameters.iter() {
            borrowed.push(obj.extract::<PyRefMut<PyTensor>>()?);
        }

        let mut tensor_refs: Vec<&mut engine::tensor::Tensor> =
            borrowed.iter_mut().map(|t| t.tensor_mut()).collect();

        let result = match &self.inner {
            OptimizerType::SGD(opt) => opt.zero_grad(tensor_refs.as_mut_slice()),
            OptimizerType::Adam(opt) => opt.zero_grad(tensor_refs.as_mut_slice()),
            OptimizerType::RMSprop(opt) => opt.zero_grad(tensor_refs.as_mut_slice()),
        };

        result.map_err(_convert_error)
    }

    /// Get learning rate
    #[getter]
    fn lr(&self) -> f64 {
        match &self.inner {
            OptimizerType::SGD(optimizer) => optimizer.learning_rate(),
            OptimizerType::Adam(optimizer) => optimizer.learning_rate(),
            OptimizerType::RMSprop(optimizer) => optimizer.learning_rate(),
        }
    }

    /// Set learning rate
    #[setter]
    fn set_lr(&mut self, lr: f64) {
        match &mut self.inner {
            OptimizerType::SGD(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::Adam(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::RMSprop(optimizer) => optimizer.set_learning_rate(lr),
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        match &self.inner {
            OptimizerType::SGD(optimizer) => format!(
                "SGD(lr={}, momentum={})",
                optimizer.learning_rate(),
                optimizer.momentum()
            ),
            OptimizerType::Adam(optimizer) => format!(
                "Adam(lr={}, betas=({}, {}), eps={})",
                optimizer.learning_rate(),
                optimizer.beta1(),
                optimizer.beta2(),
                optimizer.epsilon()
            ),
            OptimizerType::RMSprop(optimizer) => format!(
                "RMSprop(lr={}, alpha={}, eps={})",
                optimizer.learning_rate(),
                optimizer.alpha(),
                optimizer.epsilon()
            ),
        }
    }
}

impl PyOptimizer {
    pub fn from_sgd(sgd: SGD) -> Self {
        Self {
            inner: OptimizerType::SGD(sgd),
        }
    }

    pub fn from_adam(adam: Adam) -> Self {
        Self {
            inner: OptimizerType::Adam(adam),
        }
    }

    pub fn from_rmsprop(rmsprop: RMSprop) -> Self {
        Self {
            inner: OptimizerType::RMSprop(rmsprop),
        }
    }
}

/// SGD optimizer
#[pyclass(name = "SGD", extends = PyOptimizer)]
pub struct PySGD;

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    #[new]
    fn new(
        learning_rate: f64,
        momentum: Option<f64>,
        weight_decay: Option<f64>,
        nesterov: Option<bool>,
    ) -> PyResult<(Self, PyOptimizer)> {
        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);

        let sgd =
            SGD::new(learning_rate, Some(momentum), Some(weight_decay)).with_nesterov(nesterov);

        Ok((Self, PyOptimizer::from_sgd(sgd)))
    }

    /// Get momentum parameter
    #[getter]
    fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.momentum())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.weight_decay())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get nesterov flag
    #[getter]
    fn nesterov(slf: PyRef<Self>) -> PyResult<bool> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.is_nesterov())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }
}

/// Adam optimizer
#[pyclass(name = "Adam", extends = PyOptimizer)]
pub struct PyAdam;

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer
    #[new]
    #[pyo3(signature=(learning_rate, betas=None, beta1=None, beta2=None, epsilon=None, weight_decay=None))]
    fn new(
        learning_rate: f64,
        betas: Option<(f64, f64)>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> PyResult<(Self, PyOptimizer)> {
        let (beta1, beta2) = match (betas, beta1, beta2) {
            (Some((b1, b2)), _, _) => (b1, b2),
            (None, Some(b1), Some(b2)) => (b1, b2),
            (None, Some(b1), None) => (b1, 0.999),
            (None, None, Some(b2)) => (0.9, b2),
            _ => (0.9, 0.999),
        };
        let epsilon = epsilon.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let adam = Adam::new(
            learning_rate,
            Some(beta1),
            Some(beta2),
            Some(epsilon),
            Some(weight_decay),
        );

        Ok((Self, PyOptimizer::from_adam(adam)))
    }

    /// Get beta1 parameter
    #[getter]
    fn beta1(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.beta1())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get beta2 parameter
    #[getter]
    fn beta2(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.beta2())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.epsilon())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.weight_decay())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }
}

/// RMSprop optimizer
#[pyclass(name = "RMSprop", extends = PyOptimizer)]
pub struct PyRMSprop;

#[pymethods]
impl PyRMSprop {
    /// Create a new RMSprop optimizer
    #[new]
    fn new(
        learning_rate: f64,
        alpha: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
        momentum: Option<f64>,
    ) -> PyResult<(Self, PyOptimizer)> {
        let alpha = alpha.unwrap_or(0.99);
        let epsilon = epsilon.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);

        let rmsprop = RMSprop::new(
            learning_rate,
            Some(alpha),
            Some(epsilon),
            Some(weight_decay),
            Some(momentum),
        );

        Ok((Self, PyOptimizer::from_rmsprop(rmsprop)))
    }

    /// Get alpha parameter
    #[getter]
    fn alpha(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.alpha())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.epsilon())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.weight_decay())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }

    /// Get momentum parameter
    #[getter]
    fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.momentum())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Invalid optimizer type",
            ))
        }
    }
}

/// Register optimizer module with Python
pub fn register_optim_module(py: Python, parent_module: &Bound<Pyo3Module>) -> PyResult<()> {
    let optim_module = Pyo3Module::new(py, "optim")?;

    // Add optimizer classes
    optim_module.add_class::<PyOptimizer>()?;
    optim_module.add_class::<PySGD>()?;
    optim_module.add_class::<PyAdam>()?;
    optim_module.add_class::<PyRMSprop>()?;

    parent_module.add_submodule(&optim_module)?;
    Ok(())
}
