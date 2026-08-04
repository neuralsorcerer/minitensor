// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::optim::{Adagrad, Adam, AdamW, Lion, NAdam, Optimizer, RMSprop, SGD};
use engine::{autograd, tensor::Tensor};
use pyo3::Py;
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule as Pyo3Module};

/// Base class for optimizers
/// Python spells its booleans `True`/`False`; Rust's `Display` gives
/// `true`/`false`, which is not valid Python in a `__repr__`.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    inner: OptimizerType,
    parameters: Vec<Py<PyAny>>,
}

enum OptimizerType {
    Sgd(SGD),
    Adam(Adam),
    AdamW(AdamW),
    RMSprop(RMSprop),
    Adagrad(Adagrad),
    NAdam(NAdam),
    Lion(Lion),
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
                OptimizerType::Sgd(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::Adam(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::AdamW(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::RMSprop(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::Adagrad(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::NAdam(opt) => opt.step(tensor_refs.as_mut_slice()),
                OptimizerType::Lion(opt) => opt.step(tensor_refs.as_mut_slice()),
            }
            .map_err(_convert_error)?;

            // Consume only this optimizer's gradients, not the whole graph.
            //
            // This used to call `clear_graph()`, which discarded every stored
            // gradient -- including those belonging to a *different* optimizer
            // over a different parameter group. Its `step()` then found nothing
            // to apply and silently did nothing: no error, no warning, the
            // parameters simply never moved. Two optimizers over disjoint
            // groups is an ordinary arrangement (a lower learning rate for a
            // pretrained encoder than for a fresh head, a generator and a
            // discriminator sharing one loss).
            //
            // The wholesale clear was there to bound memory per iteration,
            // back when `backward()` marked the graph consumed without freeing
            // it. It frees the subgraph it walked now, and holds interior
            // gradients for only one pass, so what is left to release here is
            // just what this optimizer consumed.
            for tensor in &tensor_refs {
                autograd::clear_gradient(tensor);
            }
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
                OptimizerType::Sgd(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::Adam(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::AdamW(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::RMSprop(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::Adagrad(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::NAdam(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
                OptimizerType::Lion(opt) => opt.zero_grad(tensor_refs.as_mut_slice(), set),
            }
            .map_err(_convert_error)?;
        }

        Ok(())
    }

    /// Get learning rate
    #[getter]
    pub(crate) fn lr(&self) -> f64 {
        match &self.inner {
            OptimizerType::Sgd(optimizer) => optimizer.learning_rate(),
            OptimizerType::Adam(optimizer) => optimizer.learning_rate(),
            OptimizerType::AdamW(optimizer) => optimizer.learning_rate(),
            OptimizerType::RMSprop(optimizer) => optimizer.learning_rate(),
            OptimizerType::Adagrad(optimizer) => optimizer.learning_rate(),
            OptimizerType::NAdam(optimizer) => optimizer.learning_rate(),
            OptimizerType::Lion(optimizer) => optimizer.learning_rate(),
        }
    }

    /// Set learning rate
    #[setter]
    pub(crate) fn set_lr(&mut self, lr: f64) {
        match &mut self.inner {
            OptimizerType::Sgd(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::Adam(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::AdamW(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::RMSprop(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::Adagrad(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::NAdam(optimizer) => optimizer.set_learning_rate(lr),
            OptimizerType::Lion(optimizer) => optimizer.set_learning_rate(lr),
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        match &self.inner {
            OptimizerType::Sgd(optimizer) => format!(
                "SGD(lr={:?}, momentum={:?}, dampening={:?}, weight_decay={:?}, nesterov={})",
                optimizer.learning_rate(),
                optimizer.momentum(),
                optimizer.dampening(),
                optimizer.weight_decay(),
                py_bool(optimizer.is_nesterov())
            ),
            OptimizerType::Adam(optimizer) => format!(
                "Adam(lr={:?}, betas=({:?}, {:?}), eps={:?}, weight_decay={:?}, amsgrad={}, decoupled_weight_decay={})",
                optimizer.learning_rate(),
                optimizer.beta1(),
                optimizer.beta2(),
                optimizer.epsilon(),
                optimizer.weight_decay(),
                py_bool(optimizer.is_amsgrad()),
                py_bool(optimizer.is_decoupled_weight_decay())
            ),
            OptimizerType::AdamW(optimizer) => format!(
                "AdamW(lr={:?}, betas=({}, {}), eps={:?}, weight_decay={:?})",
                optimizer.learning_rate(),
                optimizer.beta1(),
                optimizer.beta2(),
                optimizer.epsilon(),
                optimizer.weight_decay()
            ),
            OptimizerType::NAdam(optimizer) => format!(
                "NAdam(lr={:?}, betas=({}, {}), eps={:?}, momentum_decay={:?})",
                optimizer.learning_rate(),
                optimizer.beta1(),
                optimizer.beta2(),
                optimizer.epsilon(),
                optimizer.momentum_decay()
            ),
            OptimizerType::Adagrad(optimizer) => format!(
                "Adagrad(lr={:?}, lr_decay={:?}, eps={:?})",
                optimizer.learning_rate(),
                optimizer.lr_decay(),
                optimizer.epsilon()
            ),
            OptimizerType::RMSprop(optimizer) => format!(
                "RMSprop(lr={:?}, alpha={:?}, eps={:?}, weight_decay={:?}, momentum={:?}, centered={})",
                optimizer.learning_rate(),
                optimizer.alpha(),
                optimizer.epsilon(),
                optimizer.weight_decay(),
                optimizer.momentum(),
                py_bool(optimizer.is_centered())
            ),
            OptimizerType::Lion(optimizer) => format!(
                "Lion(lr={:?}, betas=({}, {}), weight_decay={:?})",
                optimizer.learning_rate(),
                optimizer.beta1(),
                optimizer.beta2(),
                optimizer.weight_decay()
            ),
        }
    }
}

impl PyOptimizer {
    fn from_sgd(sgd: SGD, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::Sgd(sgd),
            parameters,
        }
    }

    fn from_adam(adam: Adam, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::Adam(adam),
            parameters,
        }
    }

    fn from_adamw(adamw: AdamW, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::AdamW(adamw),
            parameters,
        }
    }

    fn from_rmsprop(rmsprop: RMSprop, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::RMSprop(rmsprop),
            parameters,
        }
    }

    fn from_nadam(nadam: NAdam, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::NAdam(nadam),
            parameters,
        }
    }

    fn from_adagrad(adagrad: Adagrad, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::Adagrad(adagrad),
            parameters,
        }
    }

    fn from_lion(lion: Lion, parameters: Vec<Py<PyAny>>) -> Self {
        Self {
            inner: OptimizerType::Lion(lion),
            parameters,
        }
    }
}

fn ensure_tensor_like(value: &Bound<PyAny>) -> PyResult<()> {
    if value.extract::<PyRef<PyTensor>>().is_ok() {
        return Ok(());
    }

    let py = value.py();
    if let Ok(inner) = value.getattr(intern!(py, "_tensor"))
        && inner.extract::<PyRef<PyTensor>>().is_ok()
    {
        return Ok(());
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
    Ok(inner.extract::<PyRefMut<PyTensor>>()?)
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

fn resolve_betas(
    betas: Option<(f64, f64)>,
    beta1: Option<f64>,
    beta2: Option<f64>,
) -> PyResult<(f64, f64)> {
    resolve_betas_with_defaults(betas, beta1, beta2, (0.9, 0.999))
}

/// Resolve beta coefficients against optimizer-specific defaults. Lion, for
/// example, defaults to (0.9, 0.99) rather than Adam's (0.9, 0.999).
fn resolve_betas_with_defaults(
    betas: Option<(f64, f64)>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    defaults: (f64, f64),
) -> PyResult<(f64, f64)> {
    if betas.is_some() && (beta1.is_some() || beta2.is_some()) {
        return Err(PyTypeError::new_err(
            "specify either betas tuple or beta1/beta2, not both",
        ));
    }

    let (beta1, beta2) = if let Some((b1, b2)) = betas {
        (b1, b2)
    } else {
        match (beta1, beta2) {
            (Some(b1), Some(b2)) => (b1, b2),
            (None, None) => defaults,
            _ => {
                return Err(PyTypeError::new_err(
                    "both beta1 and beta2 must be provided",
                ));
            }
        }
    };

    validate_beta("beta1", beta1)?;
    validate_beta("beta2", beta2)?;

    Ok((beta1, beta2))
}

/// SGD optimizer
#[pyclass(name = "SGD", extends = PyOptimizer)]
pub struct PySGD;

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer
    #[new]
    #[pyo3(signature = (parameters, lr, momentum=None, dampening=None, weight_decay=None, nesterov=None))]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        momentum: Option<f64>,
        dampening: Option<f64>,
        weight_decay: Option<f64>,
        nesterov: Option<bool>,
    ) -> PyResult<PyClassInitializer<Self>> {
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

        let dampening = dampening.unwrap_or(0.0);

        let nesterov = nesterov.unwrap_or(false);
        if nesterov && momentum <= 0.0 {
            return Err(PyValueError::new_err(
                "Nesterov momentum requires a positive momentum value.",
            ));
        }
        // Nesterov's lookahead is `grad + momentum * buf`, which is only the
        // correct extrapolation when `buf` accumulated the full gradient.
        // PyTorch rejects the combination for the same reason.
        if nesterov && dampening != 0.0 {
            return Err(PyValueError::new_err(
                "Nesterov momentum requires zero dampening.",
            ));
        }

        let sgd = SGD::new(lr, Some(momentum), Some(weight_decay))
            .with_dampening(dampening)
            .with_nesterov(nesterov);

        Ok(PyClassInitializer::from(PyOptimizer::from_sgd(sgd, params)).add_subclass(Self))
    }

    /// Get momentum parameter
    #[getter]
    fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Sgd(sgd) = &optimizer.inner {
            Ok(sgd.momentum())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get momentum dampening parameter
    #[getter]
    fn dampening(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Sgd(sgd) = &optimizer.inner {
            Ok(sgd.dampening())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Sgd(sgd) = &optimizer.inner {
            Ok(sgd.weight_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get nesterov flag
    #[getter]
    fn nesterov(slf: PyRef<Self>) -> PyResult<bool> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Sgd(sgd) = &optimizer.inner {
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
            lr=1e-3,
            betas=None,
            beta1=None,
            beta2=None,
            epsilon=1e-8,
            weight_decay=0.0,
            amsgrad=false
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        betas: Option<(f64, f64)>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
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
        let (beta1, beta2) = resolve_betas(betas, beta1, beta2)?;

        // The engine has carried `with_amsgrad` (and a tested `v_hat` update)
        // since the start; nothing bound it, so the max-second-moment variant
        // was unreachable from Python.
        let adam = Adam::new(
            lr,
            Some(beta1),
            Some(beta2),
            Some(epsilon),
            Some(weight_decay),
        )
        .with_amsgrad(amsgrad);

        Ok(PyClassInitializer::from(PyOptimizer::from_adam(adam, params)).add_subclass(Self))
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

    /// Whether the AMSGrad variant is in use
    #[getter]
    fn amsgrad(slf: PyRef<Self>) -> PyResult<bool> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adam(adam) = &optimizer.inner {
            Ok(adam.is_amsgrad())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }
}

/// AdamW optimizer
#[pyclass(name = "AdamW", extends = PyOptimizer)]
pub struct PyAdamW;

#[pymethods]
impl PyAdamW {
    /// Create a new AdamW optimizer
    #[new]
    #[pyo3(
        signature = (
            parameters,
            lr=1e-3,
            betas=None,
            beta1=None,
            beta2=None,
            epsilon=1e-8,
            weight_decay=0.01
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        betas: Option<(f64, f64)>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: f64,
        weight_decay: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
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
        let (beta1, beta2) = resolve_betas(betas, beta1, beta2)?;

        let adamw = AdamW::new(
            lr,
            Some(beta1),
            Some(beta2),
            Some(epsilon),
            Some(weight_decay),
        );

        Ok(PyClassInitializer::from(PyOptimizer::from_adamw(adamw, params)).add_subclass(Self))
    }

    /// Get beta1 parameter
    #[getter]
    fn beta1(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::AdamW(adamw) = &optimizer.inner {
            Ok(adamw.beta1())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get beta2 parameter
    #[getter]
    fn beta2(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::AdamW(adamw) = &optimizer.inner {
            Ok(adamw.beta2())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::AdamW(adamw) = &optimizer.inner {
            Ok(adamw.epsilon())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::AdamW(adamw) = &optimizer.inner {
            Ok(adamw.weight_decay())
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
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        alpha: f64,
        epsilon: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
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

        Ok(PyClassInitializer::from(PyOptimizer::from_rmsprop(rmsprop, params)).add_subclass(Self))
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

/// NAdam optimizer (Dozat, 2016) — Adam with Nesterov momentum.
#[pyclass(name = "NAdam", extends = PyOptimizer)]
pub struct PyNAdam;

#[pymethods]
impl PyNAdam {
    /// Create a new NAdam optimizer
    #[new]
    #[pyo3(signature = (parameters, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, momentum_decay=0.004))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
        momentum_decay: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(PyValueError::new_err(
                "Beta coefficients must be in the range [0, 1).",
            ));
        }
        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("Epsilon must be positive."));
        }
        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }
        if momentum_decay < 0.0 {
            return Err(PyValueError::new_err(
                "momentum_decay must be non-negative.",
            ));
        }

        let params = collect_parameters(parameters)?;
        let nadam = NAdam::new(
            lr,
            Some(beta1),
            Some(beta2),
            Some(epsilon),
            Some(weight_decay),
            Some(momentum_decay),
        );

        Ok(PyClassInitializer::from(PyOptimizer::from_nadam(nadam, params)).add_subclass(Self))
    }

    /// Get the first-moment decay rate
    #[getter]
    fn beta1(slf: PyRef<Self>) -> PyResult<f64> {
        match &slf.as_ref().inner {
            OptimizerType::NAdam(o) => Ok(o.beta1()),
            _ => Err(PyRuntimeError::new_err("Invalid optimizer type")),
        }
    }

    /// Get the second-moment decay rate
    #[getter]
    fn beta2(slf: PyRef<Self>) -> PyResult<f64> {
        match &slf.as_ref().inner {
            OptimizerType::NAdam(o) => Ok(o.beta2()),
            _ => Err(PyRuntimeError::new_err("Invalid optimizer type")),
        }
    }

    /// Get epsilon
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        match &slf.as_ref().inner {
            OptimizerType::NAdam(o) => Ok(o.epsilon()),
            _ => Err(PyRuntimeError::new_err("Invalid optimizer type")),
        }
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        match &slf.as_ref().inner {
            OptimizerType::NAdam(o) => Ok(o.weight_decay()),
            _ => Err(PyRuntimeError::new_err("Invalid optimizer type")),
        }
    }

    /// Get the momentum-schedule decay
    #[getter]
    fn momentum_decay(slf: PyRef<Self>) -> PyResult<f64> {
        match &slf.as_ref().inner {
            OptimizerType::NAdam(o) => Ok(o.momentum_decay()),
            _ => Err(PyRuntimeError::new_err("Invalid optimizer type")),
        }
    }
}

/// Adagrad optimizer
#[pyclass(name = "Adagrad", extends = PyOptimizer)]
pub struct PyAdagrad;

#[pymethods]
impl PyAdagrad {
    /// Create a new Adagrad optimizer
    #[new]
    #[pyo3(
        signature = (
            parameters,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            initial_accumulator_value=0.0,
            epsilon=1e-10
        )
    )]
    fn new(
        _py: Python,
        parameters: &Bound<PyAny>,
        lr: f64,
        lr_decay: f64,
        weight_decay: f64,
        initial_accumulator_value: f64,
        epsilon: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }
        if lr_decay < 0.0 {
            return Err(PyValueError::new_err("lr_decay must be non-negative."));
        }
        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }
        if initial_accumulator_value < 0.0 {
            return Err(PyValueError::new_err(
                "initial_accumulator_value must be non-negative.",
            ));
        }
        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("Epsilon must be positive."));
        }

        let params = collect_parameters(parameters)?;
        let adagrad = Adagrad::new(
            lr,
            Some(lr_decay),
            Some(weight_decay),
            Some(initial_accumulator_value),
            Some(epsilon),
        );

        Ok(PyClassInitializer::from(PyOptimizer::from_adagrad(adagrad, params)).add_subclass(Self))
    }

    /// Get the learning-rate decay coefficient
    #[getter]
    fn lr_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adagrad(adagrad) = &optimizer.inner {
            Ok(adagrad.lr_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get epsilon parameter
    #[getter]
    fn epsilon(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adagrad(adagrad) = &optimizer.inner {
            Ok(adagrad.epsilon())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adagrad(adagrad) = &optimizer.inner {
            Ok(adagrad.weight_decay())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get the value new accumulators start at
    #[getter]
    fn initial_accumulator_value(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Adagrad(adagrad) = &optimizer.inner {
            Ok(adagrad.initial_accumulator_value())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }
}

/// Lion optimizer (Chen et al., 2023) — sign-momentum update with decoupled
/// weight decay. Half the optimizer state of Adam and often a stronger
/// large-model optimizer; because updates are sign-based, use a smaller learning
/// rate (≈3-10×) and a larger weight decay than AdamW.
#[pyclass(name = "Lion", extends = PyOptimizer)]
pub struct PyLion;

#[pymethods]
impl PyLion {
    /// Create a new Lion optimizer
    #[new]
    #[pyo3(
        signature = (
            parameters,
            lr=1e-4,
            betas=None,
            beta1=None,
            beta2=None,
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
        weight_decay: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        if lr <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive."));
        }

        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("Weight decay must be non-negative."));
        }

        let params = collect_parameters(parameters)?;
        // Lion's paper defaults are (0.9, 0.99), not Adam's (0.9, 0.999).
        let (beta1, beta2) = resolve_betas_with_defaults(betas, beta1, beta2, (0.9, 0.99))?;

        let lion = Lion::new(lr, Some(beta1), Some(beta2), Some(weight_decay));

        Ok(PyClassInitializer::from(PyOptimizer::from_lion(lion, params)).add_subclass(Self))
    }

    /// Get beta1 parameter
    #[getter]
    fn beta1(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Lion(lion) = &optimizer.inner {
            Ok(lion.beta1())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get beta2 parameter
    #[getter]
    fn beta2(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Lion(lion) = &optimizer.inner {
            Ok(lion.beta2())
        } else {
            Err(PyRuntimeError::new_err("Invalid optimizer type"))
        }
    }

    /// Get weight decay parameter
    #[getter]
    fn weight_decay(slf: PyRef<Self>) -> PyResult<f64> {
        let optimizer = slf.as_ref();
        if let OptimizerType::Lion(lion) = &optimizer.inner {
            Ok(lion.weight_decay())
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
    optim_module.add_class::<PyAdamW>()?;
    optim_module.add_class::<PyRMSprop>()?;
    optim_module.add_class::<PyAdagrad>()?;
    optim_module.add_class::<PyNAdam>()?;
    optim_module.add_class::<PyLion>()?;

    crate::lr_scheduler::register(&optim_module)?;

    parent_module.add_submodule(&optim_module)?;
    Ok(())
}
