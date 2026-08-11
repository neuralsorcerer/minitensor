// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient clipping and inspection.
//!
//! `engine::optim::GradientUtils` implements all of this -- norm computation,
//! clipping by norm, clipping by value, and the "does anything have a gradient"
//! queries -- with the parallel element loops and the graph-vs-tensor gradient
//! resolution already worked out. None of it had a binding, so a Python user
//! training anything that needed clipped gradients had to reach into `.grad`
//! and rescale by hand.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::optim::GradientUtils;
use engine::tensor::Tensor;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule as Pyo3Module};

/// Collect an iterable of tensors, accepting the `_tensor`-wrapping objects the
/// rest of the API accepts. Unlike the optimizer's version an empty sequence is
/// fine: clipping nothing is a no-op, not a mistake.
fn collect(parameters: &Bound<PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let mut collected = Vec::new();
    for item in PyIterator::from_object(parameters)? {
        collected.push(item?.unbind());
    }
    Ok(collected)
}

fn borrow_mut<'py>(py: Python<'py>, value: &'py Py<PyAny>) -> PyResult<PyRefMut<'py, PyTensor>> {
    let bound = value.bind(py);
    if let Ok(tensor) = bound.extract::<PyRefMut<PyTensor>>() {
        return Ok(tensor);
    }
    let inner = bound
        .getattr(intern!(py, "_tensor"))
        .map_err(|_| PyTypeError::new_err("parameters must be Tensor instances"))?;
    Ok(inner.extract::<PyRefMut<PyTensor>>()?)
}

/// Run `body` with every parameter borrowed mutably at once, which is what the
/// engine's slice-taking API needs.
fn with_parameters<R>(
    py: Python<'_>,
    parameters: &Bound<PyAny>,
    body: impl FnOnce(&mut [&mut Tensor]) -> PyResult<R>,
) -> PyResult<R> {
    let collected = collect(parameters)?;
    let mut borrowed: Vec<PyRefMut<PyTensor>> = Vec::with_capacity(collected.len());
    for value in &collected {
        borrowed.push(borrow_mut(py, value)?);
    }
    let mut refs: Vec<&mut Tensor> = borrowed.iter_mut().map(|t| t.tensor_mut()).collect();
    body(refs.as_mut_slice())
}

/// Scale gradients in place so their combined L2 norm is at most `max_norm`.
///
/// Returns the total norm *before* clipping, so a training loop can log it. Parameters without a gradient are skipped. Only float
/// gradients participate.
#[pyfunction]
#[pyo3(name = "clip_grad_norm_", signature = (parameters, max_norm))]
pub fn clip_grad_norm(py: Python<'_>, parameters: &Bound<PyAny>, max_norm: f64) -> PyResult<f64> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(PyValueError::new_err(
            "max_norm must be positive and finite",
        ));
    }
    with_parameters(py, parameters, |refs| {
        GradientUtils::clip_grad_norm(refs, max_norm).map_err(_convert_error)
    })
}

/// Clamp every gradient element in place.
///
/// With only `clip_value` the range is `[-clip_value, clip_value]`.
/// `min_value`/`max_value` give the asymmetric form the engine also
/// supports.
#[pyfunction]
#[pyo3(name = "clip_grad_value_", signature = (parameters, clip_value=None, min_value=None, max_value=None))]
pub fn clip_grad_value(
    py: Python<'_>,
    parameters: &Bound<PyAny>,
    clip_value: Option<f64>,
    min_value: Option<f64>,
    max_value: Option<f64>,
) -> PyResult<()> {
    let (low, high) = match (clip_value, min_value, max_value) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err(PyValueError::new_err(
                "pass clip_value or min_value/max_value, not both",
            ));
        }
        (Some(value), None, None) => {
            if !value.is_finite() || value <= 0.0 {
                return Err(PyValueError::new_err(
                    "clip_value must be positive and finite",
                ));
            }
            (-value, value)
        }
        (None, Some(low), Some(high)) => (low, high),
        _ => {
            return Err(PyValueError::new_err(
                "pass either clip_value, or both min_value and max_value",
            ));
        }
    };

    if !low.is_finite() || !high.is_finite() || low > high {
        return Err(PyValueError::new_err(
            "min_value must be finite and no greater than max_value",
        ));
    }

    with_parameters(py, parameters, |refs| {
        GradientUtils::clip_grad_value(refs, low, high).map_err(_convert_error)
    })
}

/// The combined L2 norm of the parameters' gradients, without modifying them.
#[pyfunction]
#[pyo3(name = "grad_norm", signature = (parameters))]
pub fn grad_norm(py: Python<'_>, parameters: &Bound<PyAny>) -> PyResult<f64> {
    with_parameters(py, parameters, |refs| {
        let immutable: Vec<&Tensor> = refs.iter().map(|t| &**t).collect();
        GradientUtils::compute_grad_norm(&immutable).map_err(_convert_error)
    })
}

/// How many of the parameters currently hold a gradient.
#[pyfunction]
#[pyo3(name = "count_parameters_with_gradients", signature = (parameters))]
pub fn count_parameters_with_gradients(
    py: Python<'_>,
    parameters: &Bound<PyAny>,
) -> PyResult<usize> {
    with_parameters(py, parameters, |refs| {
        let immutable: Vec<&Tensor> = refs.iter().map(|t| &**t).collect();
        Ok(GradientUtils::count_parameters_with_gradients(&immutable))
    })
}

pub fn register(module: &Bound<'_, Pyo3Module>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(clip_grad_norm, module)?)?;
    module.add_function(wrap_pyfunction!(clip_grad_value, module)?)?;
    module.add_function(wrap_pyfunction!(grad_norm, module)?)?;
    module.add_function(wrap_pyfunction!(count_parameters_with_gradients, module)?)?;
    Ok(())
}
