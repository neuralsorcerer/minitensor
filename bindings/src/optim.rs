// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::optim::{Adam, Optimizer, RMSprop, SGD};
use pyo3::prelude::*;
use pyo3::types::PyModule as Pyo3Module;

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
    /// Perform a single optimization step (placeholder)
    fn step(&mut self) -> PyResult<()> {
        // For now, this is a placeholder implementation
        // In a full implementation, we would need to handle parameter updates
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Optimizer step not yet implemented",
        ))
    }

    /// Zero all gradients (placeholder)
    fn zero_grad(&self) -> PyResult<()> {
        // For now, this is a placeholder implementation
        // In a full implementation, we would zero gradients for all tracked parameters
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Optimizer zero_grad not yet implemented",
        ))
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
    fn new(
        learning_rate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> PyResult<(Self, PyOptimizer)> {
        let beta1 = beta1.unwrap_or(0.9);
        let beta2 = beta2.unwrap_or(0.999);
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
pub fn register_optim_module(py: Python, parent_module: &Pyo3Module) -> PyResult<()> {
    let optim_module = Pyo3Module::new(py, "optim")?;

    // Add optimizer classes
    optim_module.add_class::<PyOptimizer>()?;
    optim_module.add_class::<PySGD>()?;
    optim_module.add_class::<PyAdam>()?;
    optim_module.add_class::<PyRMSprop>()?;

    parent_module.add_submodule(optim_module)?;
    Ok(())
}
