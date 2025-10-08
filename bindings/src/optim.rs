// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::optim::{Adam, Optimizer, RMSprop, SGD};
use engine::{autograd, tensor::Tensor};
use pyo3::Py;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule as Pyo3Module};

/// Base class for optimizers
#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    inner: OptimizerType,
    parameters: Vec<Py<PyAny>>,
}

enum OptimizerType {
    SGD(SGD),
    Adam(Adam),
    RMSprop(RMSprop),
}

#[pymethods]
impl PyOptimizer {
    /// Perform a single optimization step using the tracked parameters.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.parameters.is_empty() {
            return Err(PyValueError::new_err("No parameters to optimize."));
        }

        {
            let mut borrowed: Vec<PyRefMut<PyTensor>> = Vec::with_capacity(self.parameters.len());
            for value in &self.parameters {
                borrowed.push(borrow_tensor_mut(py, value)?);
            }

            let mut tensor_refs: Vec<&mut Tensor> = borrowed
                .iter_mut()
                .map(|tensor| tensor.tensor_mut())
                .collect();

            match &mut self.inner {
                OptimizerType::SGD(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::Adam(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::RMSprop(opt) => opt.step(tensor_refs.as_mut_slice()),
            }
            .map_err(_convert_error)?;
        }

        if let Err(e) = autograd::clear_graph() {
            return Err(_convert_error(e));
        }
        Ok(())
    }

    /// Zero out gradients for the tracked parameters.
    #[pyo3(signature = (set_to_none=None))]
    fn zero_grad(&mut self, py: Python<'_>, set_to_none: Option<bool>) -> PyResult<()> {
        if self.parameters.is_empty() {
            return Err(PyValueError::new_err("No parameters to optimize."));
        }

        let set = set_to_none.unwrap_or(false);

        {
            let mut borrowed: Vec<PyRefMut<PyTensor>> = Vec::with_capacity(self.parameters.len());
            for value in &self.parameters {
                borrowed.push(borrow_tensor_mut(py, value)?);
            }

            let mut tensor_refs: Vec<&mut Tensor> = borrowed
                .iter_mut()
                .map(|tensor| tensor.tensor_mut())
                .collect();

            match &mut self.inner {
                OptimizerType::SGD(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::Adam(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::RMSprop(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
            }
            .map_err(_convert_error)?;
        }

        Ok(())
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
    fn from_sgd(sgd: SGD, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::SGD(sgd),
            parameters,
        }
    }

    fn from_adam(adam: Adam, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::Adam(adam),
            parameters,
        }
    }

    fn from_rmsprop(rmsprop: RMSprop, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::RMSprop(rmsprop),
            parameters,
        }
    }
}

fn ensure_tensor_like(value: &Bound<PyAny>) -> PyResult<()> {
    if value.extract::<PyRef<PyTensor>>().is_ok() {
        return Ok(());
    }

    let py = value.py();
    if let Ok(inner) = value.getattr(intern!(py, "_tensor")) {
        if inner.extract::<PyRef<PyTensor>>().is_ok() {
            return Ok(());
        }
    }

    Err(PyTypeError::new_err(
        "optimizer parameters must be Tensor instances",
    ))
}

fn borrow_tensor_mut<'py>(
    py: Python<'py>,
    value: &'py Py<PyAny>,
) -> PyResult<PyRefMut<'py, PyTensor>> {
    let bound = value.bind(py);
    if let Ok(tensor) = bound.extract::<PyRefMut<PyTensor>>() {
        return Ok(tensor);
    }

    let inner = bound
        .getattr(intern!(py, "_tensor"))
        .map_err(|_| PyTypeError::new_err("optimizer parameters must be Tensor instances"))?;
    inner.extract::<PyRefMut<PyTensor>>()
}

fn collect_parameters(parameters: &Bound<PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let iterator = PyIterator::from_object(parameters)?;
    let mut collected: Vec<Py<PyAny>> = Vec::new();

    for item in iterator {
        let value = item?;
        ensure_tensor_like(&value)?;
        collected.push(value.unbind());
    }

    if collected.is_empty() {
        return Err(PyValueError::new_err("No parameters to optimize."));
    }

    Ok(collected)
}

fn validate_beta(name: &str, value: f64) -> PyResult<()> {
    if !(0.0..1.0).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "{} must be in the range [0, 1).",
            name
        )));
    }
    Ok(())
}

/// SGD optimizer
#[pyclass(name = "SGD", extends = PyOptimizer)]
pub struct PySGD;

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    #[new]
    #[pyo3(signature = (parameters, lr, momentum=None, weight_decay=None, nesterov=None))]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        momentum: Option<f64>,
        weight_decay: Option<f64>,
        nesterov: Option<bool>,
    ) -> PyResult<(Self, PyOptimizer)> {
        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }

        let params = collect_parameters(parameters)?;

        let momentum = momentum.unwrap_or(0.0);
        if momentum < 0.0 {
            return Err(PyValueError::new_err("Momentum must be non-negative."));
        }

        let weight_decay = weight_decay.unwrap_or(0.0);
        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }

        let nesterov = nesterov.unwrap_or(false);
        if nesterov && momentum <= 0.0 {
            return Err(PyValueError::new_err(
                "Nesterov momentum requires a positive momentum value.",
            ));
        }

        let sgd = SGD::new(lr, Some(momentum), Some(weight_decay)).with_nesterov(nesterov);

        Ok((Self, PyOptimizer::from_sgd(sgd, params)))
    }

    /// Get momentum parameter
    #[getter]
    fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.momentum())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.weight_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get nesterov flag
    #[getter]
    fn nesterov(slf: PyRef<Self>) -> PyResult<bool> {
        let optimizer = slf.as_ref();
        if let OptimizerType::SGD(sgd) = &optimizer.inner {
            Ok(sgd.is_nesterov())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
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
    #[pyo3(
        signature = (
            parameters,
            lr,
            betas=None,
            beta1=None,
            beta2=None,
            epsilon=1e-8,
            weight_decay=0.0
        )
    )]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        betas: Option<(f64, f64)>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: f64,
        weight_decay: f64,
    ) -> PyResult<(Self, PyOptimizer)> {
        if let Some(_) = betas {
            if beta1.is_some() || beta2.is_some() {
                return Err(PyTypeError::new_err(
                    "specify either betas tuple or beta1/beta2, not both",
                ));
            }
        }

        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }

        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("Epsilon must be positive."));
        }

        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }

        let params = collect_parameters(parameters)?;

        let (beta1, beta2) = if let Some((b1, b2)) = betas {
            (b1, b2)
        } else {
            match (beta1, beta2) {
                (Some(b1), Some(b2)) => (b1, b2),
                (None, None) => (0.9, 0.999),
                _ => {
                    return Err(PyTypeError::new_err(
                        "both beta1 and beta2 must be provided",
                    ));
                }
            }
        };

        validate_beta("beta1", beta1)?;
        validate_beta("beta2", beta2)?;

        let adam = Adam::new(
            lr,
            Some(beta1),
            Some(beta2),
            Some(epsilon),
            Some(weight_decay),
        );

        Ok((Self, PyOptimizer::from_adam(adam, params)))
    }

    /// Get beta1 parameter
    #[getter]
    fn beta1(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.beta1())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get beta2 parameter
    #[getter]
    fn beta2(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.beta2())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.epsilon())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.weight_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
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
    #[pyo3(
        signature = (
            parameters,
            lr,
            alpha=0.99,
            epsilon=1e-8,
            weight_decay=0.0,
            momentum=0.0,
            centered=false
        )
    )]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        alpha: f64,
        epsilon: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> PyResult<(Self, PyOptimizer)> {
        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }

        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err("Alpha must be in the range [0, 1]."));
        }

        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("Epsilon must be positive."));
        }

        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }

        if momentum < 0.0 {
            return Err(PyValueError::new_err("Momentum must be non-negative."));
        }

        let params = collect_parameters(parameters)?;

        let rmsprop = RMSprop::new(
            lr,
            Some(alpha),
            Some(epsilon),
            Some(weight_decay),
            Some(momentum),
        )
        .with_centered(centered);

        Ok((Self, PyOptimizer::from_rmsprop(rmsprop, params)))
    }

    /// Get alpha parameter
    #[getter]
    fn alpha(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.alpha())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.epsilon())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.weight_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get momentum parameter
    #[getter]
    fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::RMSprop(rmsprop) = &optimizer.inner {
            Ok(rmsprop.momentum())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
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
