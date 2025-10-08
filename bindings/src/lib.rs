// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod custom_ops;
mod debug;
mod device;
mod dtype;
mod error;
mod functional;
mod nn;
mod numpy_compat;
mod optim;
mod plugins;
mod serialization;
mod tensor;

use device::PyDevice;
use error::_convert_error;
use tensor::{PyTensor, ShapeSequence};

/// Python module for minitensor core
#[pymodule]
fn _core(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add core classes
    m.add_class::<PyTensor>()?;
    m.add_class::<ShapeSequence>()?;
    m.add_class::<PyDevice>()?;

    // Register submodules
    nn::register_nn_module(py, m)?;
    optim::register_optim_module(py, m)?;

    let functional_module = PyModule::new(py, "functional")?;
    functional::register_functional_module(py, &functional_module)?;
    m.add_submodule(&functional_module)?;

    // Add debugging utilities
    let debug_module = PyModule::new(py, "debug")?;
    debug::init_debug_module(py, &debug_module)?;
    m.add_submodule(&debug_module)?;

    // Add NumPy compatibility functions
    let numpy_module = PyModule::new(py, "numpy_compat")?;
    numpy_compat::numpy_compat(py, &numpy_module)?;
    m.add_submodule(&numpy_module)?;

    // Add custom operations functions
    custom_ops::init_custom_ops_module(py, m)?;

    // Add plugin system
    let plugins_module = PyModule::new(py, "plugins")?;
    plugins::register_plugin_module(py, &plugins_module)?;
    m.add_submodule(&plugins_module)?;

    // Add serialization module
    serialization::register_serialization_module(py, m)?;

    // Autograd helpers
    m.add_function(wrap_pyfunction!(get_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(clear_autograd_graph, m)?)?;
    m.add_function(wrap_pyfunction!(is_autograd_graph_consumed, m)?)?;
    m.add_function(wrap_pyfunction!(mark_autograd_graph_consumed, m)?)?;

    m.add_function(wrap_pyfunction!(get_default_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_dtype, m)?)?;

    Ok(())
}

#[pyfunction]
fn get_gradient(tensor: &PyTensor) -> PyResult<Option<PyTensor>> {
    Ok(engine::autograd::get_gradient(tensor.tensor()).map(PyTensor::from_tensor))
}

#[pyfunction]
fn clear_autograd_graph() -> PyResult<()> {
    engine::autograd::clear_graph().map_err(_convert_error)
}

#[pyfunction]
fn is_autograd_graph_consumed() -> PyResult<bool> {
    Ok(engine::autograd::is_graph_consumed())
}

#[pyfunction]
fn mark_autograd_graph_consumed() -> PyResult<()> {
    engine::autograd::mark_graph_consumed();
    Ok(())
}

#[pyfunction]
fn get_default_dtype() -> PyResult<String> {
    Ok(dtype::get_default_dtype())
}

#[pyfunction]
fn set_default_dtype(dtype: &str) -> PyResult<()> {
    dtype::set_default_dtype(dtype)
}
