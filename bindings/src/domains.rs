// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python bindings for the domain kernels in [`engine::ops::domains`].
//!
//! Thin by design: argument checking that Python users expect (a readable
//! error for a bad option kind, a named gate) and nothing else. Everything
//! substantive — the fused arithmetic, the analytic derivatives — is in the
//! engine, where it is testable without a Python interpreter.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::ops::domains::{finance, quantum};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn option_kind(kind: &str) -> PyResult<finance::OptionKind> {
    match kind.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(finance::OptionKind::Call),
        "put" | "p" => Ok(finance::OptionKind::Put),
        other => Err(PyValueError::new_err(format!(
            "option kind must be 'call' or 'put', got {other:?}"
        ))),
    }
}

/// Black-Scholes price for European options, elementwise.
#[pyfunction]
#[pyo3(
    signature = (spot, strike, rate, vol, time, kind="call"),
    text_signature = "(spot, strike, rate, vol, time, kind='call')"
)]
fn black_scholes(
    spot: &PyTensor,
    strike: &PyTensor,
    rate: &PyTensor,
    vol: &PyTensor,
    time: &PyTensor,
    kind: &str,
) -> PyResult<PyTensor> {
    finance::black_scholes(
        spot.tensor(),
        strike.tensor(),
        rate.tensor(),
        vol.tensor(),
        time.tensor(),
        option_kind(kind)?,
    )
    .map(PyTensor::from_tensor)
    .map_err(_convert_error)
}

/// The volatility that reproduces an observed price. NaN where none does.
///
/// Eight parameters, because Black-Scholes has five and a solver has three.
/// Grouping them into a struct would only move the same eight names to the
/// call site, where they would no longer be keyword arguments.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    signature = (price, spot, strike, rate, time, kind="call", tolerance=1e-10, max_iterations=100),
    text_signature = "(price, spot, strike, rate, time, kind='call', tolerance=1e-10, max_iterations=100)"
)]
fn implied_volatility(
    price: &PyTensor,
    spot: &PyTensor,
    strike: &PyTensor,
    rate: &PyTensor,
    time: &PyTensor,
    kind: &str,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PyTensor> {
    finance::implied_volatility(
        price.tensor(),
        spot.tensor(),
        strike.tensor(),
        rate.tensor(),
        time.tensor(),
        option_kind(kind)?,
        tolerance,
        max_iterations,
    )
    .map(PyTensor::from_tensor)
    .map_err(_convert_error)
}

/// Resolve a gate given either a name or eight reals.
///
/// The named form covers the gates every circuit uses and spares the caller
/// spelling out `1/sqrt(2)` four times; the explicit form takes
/// `[a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im]` for an arbitrary
/// `[[a, b], [c, d]]`.
fn resolve_gate(gate: &Bound<PyAny>) -> PyResult<quantum::Gate1> {
    if let Ok(name) = gate.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "h" | "hadamard" => Ok(quantum::HADAMARD),
            "x" | "pauli_x" | "not" => Ok(quantum::PAULI_X),
            "z" | "pauli_z" => Ok(quantum::PAULI_Z),
            other => Err(PyValueError::new_err(format!(
                "unknown gate {other:?}; known names are 'h', 'x', 'z', or pass eight reals"
            ))),
        };
    }
    let values: Vec<f64> = gate.extract().map_err(|_| {
        PyValueError::new_err(
            "gate must be a name or a sequence of eight reals \
             [a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im]",
        )
    })?;
    let array: [f64; 8] = values.try_into().map_err(|v: Vec<f64>| {
        PyValueError::new_err(format!(
            "a 2x2 complex gate needs eight reals, got {}",
            v.len()
        ))
    })?;
    Ok(array)
}

/// Apply a single-qubit gate to a state vector shaped `(..., 2**q, 2)`.
#[pyfunction]
#[pyo3(text_signature = "(state, gate, qubit)")]
fn apply_gate_1q(state: &PyTensor, gate: &Bound<PyAny>, qubit: usize) -> PyResult<PyTensor> {
    quantum::apply_gate_1q(state.tensor(), &resolve_gate(gate)?, qubit)
        .map(PyTensor::from_tensor)
        .map_err(_convert_error)
}

/// Born-rule probabilities, one per basis state.
#[pyfunction]
#[pyo3(text_signature = "(state)")]
fn probabilities(state: &PyTensor) -> PyResult<PyTensor> {
    quantum::probabilities(state.tensor())
        .map(PyTensor::from_tensor)
        .map_err(_convert_error)
}

/// Marginal probabilities of the first `keep` qubits.
#[pyfunction]
#[pyo3(text_signature = "(state, keep)")]
fn prefix_trace(state: &PyTensor, keep: usize) -> PyResult<PyTensor> {
    quantum::prefix_trace(state.tensor(), keep)
        .map(PyTensor::from_tensor)
        .map_err(_convert_error)
}

/// The expectation of Pauli-Z on one qubit.
#[pyfunction]
#[pyo3(text_signature = "(state, qubit)")]
fn expect_z(state: &PyTensor, qubit: usize) -> PyResult<PyTensor> {
    quantum::expect_z(state.tensor(), qubit)
        .map(PyTensor::from_tensor)
        .map_err(_convert_error)
}

pub fn register_domains_module(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "domains")?;
    module.setattr(
        "__doc__",
        "Fused, differentiable primitives for clustering, quantitative finance \
         and quantum-inspired state vectors.",
    )?;

    let fin = PyModule::new(py, "finance")?;
    fin.setattr(
        "__doc__",
        "Black-Scholes pricing, with the Greeks as the gradient.",
    )?;
    fin.add_function(wrap_pyfunction!(black_scholes, &fin)?)?;
    fin.add_function(wrap_pyfunction!(implied_volatility, &fin)?)?;
    module.add_submodule(&fin)?;

    let qm = PyModule::new(py, "quantum")?;
    qm.setattr("__doc__", "State-vector evolution and measurement.")?;
    qm.add_function(wrap_pyfunction!(apply_gate_1q, &qm)?)?;
    qm.add_function(wrap_pyfunction!(probabilities, &qm)?)?;
    qm.add_function(wrap_pyfunction!(prefix_trace, &qm)?)?;
    qm.add_function(wrap_pyfunction!(expect_z, &qm)?)?;
    module.add_submodule(&qm)?;

    parent.add_submodule(&module)?;
    Ok(())
}
