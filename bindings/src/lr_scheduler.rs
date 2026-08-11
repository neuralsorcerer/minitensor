// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Learning-rate schedulers.
//!
//! The engine has implemented seven of these since the start
//! (`engine::optim::{ConstantLR, StepLR, ExponentialLR, CosineAnnealingLR}` and
//! `engine::optim::{LinearWarmupScheduler, PolynomialDecayScheduler,
//! MultiStepScheduler}`), all behind the [`LearningRateScheduler`] trait, and
//! none of them had a binding -- so a Python user could train but could not
//! decay a learning rate without writing the schedule by hand.
//!
//! The engine trait is `get_lr(step, base_lr) -> f64`: pure, with the step
//! counter owned by the caller. These wrappers own that counter and the base
//! learning rate, and write the result back to the optimizer.

use crate::optim::PyOptimizer;
use engine::optim::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, LearningRateScheduler, LinearWarmupScheduler,
    MultiStepScheduler, PolynomialDecayScheduler, StepLR,
};
use pyo3::Py;
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule as Pyo3Module};

/// Base class for learning-rate schedulers.
#[pyclass(name = "LRScheduler", subclass)]
pub struct PyLRScheduler {
    inner: Box<dyn LearningRateScheduler + Send + Sync>,
    optimizer: Py<PyOptimizer>,
    base_lr: f64,
    last_epoch: usize,
}

#[pymethods]
impl PyLRScheduler {
    /// Advance one step and write the new learning rate to the optimizer.
    fn step(&mut self, py: Python<'_>) -> PyResult<f64> {
        self.inner.step();
        self.last_epoch += 1;
        self.apply(py)
    }

    /// The learning rate this scheduler last wrote to the optimizer.
    fn get_last_lr(&self) -> f64 {
        self.inner.get_lr(self.last_epoch, self.base_lr)
    }

    /// The learning rate this schedule produces at `step`, without applying it.
    ///
    /// Useful for plotting a schedule before training, and for asserting on it
    /// in a test without driving an optimizer.
    fn get_lr(&self, step: usize) -> f64 {
        self.inner.get_lr(step, self.base_lr)
    }

    /// The learning rate the optimizer had when this scheduler was created.
    ///
    /// Every schedule is expressed relative to it, so changing the optimizer's
    /// `lr` afterwards does not move the schedule -- construct a new scheduler
    /// instead.
    #[getter]
    fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Number of `step()` calls so far.
    #[getter]
    fn last_epoch(&self) -> usize {
        self.last_epoch
    }

    /// Snapshot the schedule's position, so a resumed run continues it.
    ///
    /// Every schedule here is a pure function of `(last_epoch, base_lr)` --
    /// none of them accumulate -- so those two numbers are the whole of the
    /// mutable state. A plain dict, because there are no tensors to write
    /// and the caller can put it wherever the rest of their
    /// checkpoint goes.
    ///
    /// Without it, restoring a checkpoint restarted the schedule: a run that
    /// had decayed to a tenth of its base rate resumed at the base rate, and
    /// stayed a full schedule ahead of where it should have been for the rest
    /// of training.
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = PyDict::new(py);
        state.set_item("base_lr", self.base_lr)?;
        state.set_item("last_epoch", self.last_epoch)?;
        Ok(state)
    }

    /// Restore a snapshot from [`Self::state_dict`] and write the schedule's
    /// learning rate to the optimizer immediately.
    ///
    /// Applying on load matters: the optimizer this scheduler was constructed
    /// over has whatever rate it was built with, and without this the restored
    /// schedule would not take effect until the next `step()` -- one step of
    /// training at the wrong rate.
    fn load_state_dict(&mut self, py: Python<'_>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let base_lr = state
            .get_item("base_lr")?
            .ok_or_else(|| PyKeyError::new_err("scheduler state is missing 'base_lr'"))?
            .extract::<f64>()?;
        // Extracted as a signed integer so a negative one is rejected with a
        // message about the schedule rather than pyo3's OverflowError from the
        // `usize` conversion, which reads as an internal detail.
        let last_epoch = state
            .get_item("last_epoch")?
            .ok_or_else(|| PyKeyError::new_err("scheduler state is missing 'last_epoch'"))?
            .extract::<i64>()?;
        if last_epoch < 0 {
            return Err(PyValueError::new_err(format!(
                "last_epoch must not be negative, got {last_epoch}"
            )));
        }
        let last_epoch = last_epoch as usize;
        if !base_lr.is_finite() {
            return Err(PyValueError::new_err("base_lr must be finite"));
        }
        self.base_lr = base_lr;
        self.last_epoch = last_epoch;
        self.apply(py)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "LRScheduler(base_lr={}, last_epoch={}, lr={})",
            self.base_lr,
            self.last_epoch,
            self.get_last_lr()
        )
    }
}

impl PyLRScheduler {
    /// Build the base class and apply the schedule's step-0 value, so
    /// constructing a scheduler already sets the initial rate.
    fn build(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        inner: Box<dyn LearningRateScheduler + Send + Sync>,
    ) -> PyResult<Self> {
        let base_lr = optimizer.borrow(py).lr();
        let mut scheduler = Self {
            inner,
            optimizer,
            base_lr,
            last_epoch: 0,
        };
        scheduler.apply(py)?;
        Ok(scheduler)
    }

    fn apply(&mut self, py: Python<'_>) -> PyResult<f64> {
        let lr = self.inner.get_lr(self.last_epoch, self.base_lr);
        self.optimizer.borrow_mut(py).set_lr(lr);
        Ok(lr)
    }
}

fn positive(name: &str, value: usize) -> PyResult<usize> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be positive")));
    }
    Ok(value)
}

fn finite_non_negative(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(value)
}

/// Holds the learning rate constant. Useful as a no-op in a training loop that
/// always constructs a scheduler.
#[pyclass(name = "ConstantLR", extends = PyLRScheduler)]
pub struct PyConstantLR;

#[pymethods]
impl PyConstantLR {
    #[new]
    fn new(py: Python<'_>, optimizer: Py<PyOptimizer>) -> PyResult<PyClassInitializer<Self>> {
        let base = PyLRScheduler::build(py, optimizer, Box::new(ConstantLR))?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Multiplies the learning rate by `gamma` every `step_size` steps.
#[pyclass(name = "StepLR", extends = PyLRScheduler)]
pub struct PyStepLR;

#[pymethods]
impl PyStepLR {
    #[new]
    #[pyo3(signature = (optimizer, step_size, gamma=0.1))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        step_size: usize,
        gamma: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let step_size = positive("step_size", step_size)?;
        let gamma = finite_non_negative("gamma", gamma)?;
        let base = PyLRScheduler::build(py, optimizer, Box::new(StepLR::new(step_size, gamma)))?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Multiplies the learning rate by `gamma` every step.
#[pyclass(name = "ExponentialLR", extends = PyLRScheduler)]
pub struct PyExponentialLR;

#[pymethods]
impl PyExponentialLR {
    #[new]
    #[pyo3(signature = (optimizer, gamma))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        gamma: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let gamma = finite_non_negative("gamma", gamma)?;
        let base = PyLRScheduler::build(py, optimizer, Box::new(ExponentialLR::new(gamma)))?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Follows a half cosine from the base learning rate down to `eta_min`.
///
/// Reaches `eta_min` after `t_max` steps and holds it there.
#[pyclass(name = "CosineAnnealingLR", extends = PyLRScheduler)]
pub struct PyCosineAnnealingLR;

#[pymethods]
impl PyCosineAnnealingLR {
    #[new]
    #[pyo3(signature = (optimizer, t_max, eta_min=0.0))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        t_max: usize,
        eta_min: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let t_max = positive("t_max", t_max)?;
        let eta_min = finite_non_negative("eta_min", eta_min)?;
        let base = PyLRScheduler::build(
            py,
            optimizer,
            Box::new(CosineAnnealingLR::new(t_max, eta_min)),
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Ramps the learning rate linearly from 0 to its base value over
/// `warmup_steps`, then holds it.
#[pyclass(name = "LinearWarmupLR", extends = PyLRScheduler)]
pub struct PyLinearWarmupLR;

#[pymethods]
impl PyLinearWarmupLR {
    #[new]
    #[pyo3(signature = (optimizer, warmup_steps))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        warmup_steps: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        let warmup_steps = positive("warmup_steps", warmup_steps)?;
        let base = PyLRScheduler::build(
            py,
            optimizer,
            Box::new(LinearWarmupScheduler::new(warmup_steps)),
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Decays polynomially from the base learning rate to `end_lr` over
/// `decay_steps`, then holds at `end_lr`.
#[pyclass(name = "PolynomialDecayLR", extends = PyLRScheduler)]
pub struct PyPolynomialDecayLR;

#[pymethods]
impl PyPolynomialDecayLR {
    #[new]
    #[pyo3(signature = (optimizer, decay_steps, end_lr=0.0, power=1.0))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        decay_steps: usize,
        end_lr: f64,
        power: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let decay_steps = positive("decay_steps", decay_steps)?;
        let end_lr = finite_non_negative("end_lr", end_lr)?;
        if !power.is_finite() || power <= 0.0 {
            return Err(PyValueError::new_err("power must be positive and finite"));
        }
        let base = PyLRScheduler::build(
            py,
            optimizer,
            Box::new(PolynomialDecayScheduler::new(decay_steps, end_lr, power)),
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

/// Multiplies the learning rate by `gamma` once at each milestone step.
#[pyclass(name = "MultiStepLR", extends = PyLRScheduler)]
pub struct PyMultiStepLR;

#[pymethods]
impl PyMultiStepLR {
    #[new]
    #[pyo3(signature = (optimizer, milestones, gamma=0.1))]
    fn new(
        py: Python<'_>,
        optimizer: Py<PyOptimizer>,
        milestones: Vec<usize>,
        gamma: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        if milestones.is_empty() {
            return Err(PyValueError::new_err("milestones must not be empty"));
        }
        let gamma = finite_non_negative("gamma", gamma)?;
        let base = PyLRScheduler::build(
            py,
            optimizer,
            Box::new(MultiStepScheduler::new(milestones, gamma)),
        )?;
        Ok(PyClassInitializer::from(base).add_subclass(Self))
    }
}

pub fn register(module: &Bound<'_, Pyo3Module>) -> PyResult<()> {
    module.add_class::<PyLRScheduler>()?;
    module.add_class::<PyConstantLR>()?;
    module.add_class::<PyStepLR>()?;
    module.add_class::<PyExponentialLR>()?;
    module.add_class::<PyCosineAnnealingLR>()?;
    module.add_class::<PyLinearWarmupLR>()?;
    module.add_class::<PyPolynomialDecayLR>()?;
    module.add_class::<PyMultiStepLR>()?;
    Ok(())
}
