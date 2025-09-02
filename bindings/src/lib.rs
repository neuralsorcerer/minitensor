// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod custom_ops;
mod debug;
mod device;
mod error;
mod nn;
mod numpy_compat;
mod optim;
mod plugins;
mod serialization;
mod tensor;

use device::PyDevice;
use tensor::PyTensor;

/// Python module for minitensor core
#[pymodule]
fn _core(py: Python, m: &PyModule) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add core classes
    m.add_class::<PyTensor>()?;
    m.add_class::<PyDevice>()?;

    // Register submodules
    nn::register_nn_module(py, m)?;
    optim::register_optim_module(py, m)?;

    // Add debugging utilities
    let debug_module = PyModule::new(py, "debug")?;
    debug::init_debug_module(py, debug_module)?;
    m.add_submodule(debug_module)?;

    // Add NumPy compatibility functions
    let numpy_module = PyModule::new(py, "numpy_compat")?;
    numpy_compat::numpy_compat(py, numpy_module)?;
    m.add_submodule(numpy_module)?;

    // Add custom operations functions
    custom_ops::init_custom_ops_module(py, m)?;

    // Add plugin system
    let plugins_module = PyModule::new(py, "plugins")?;
    plugins::register_plugin_module(py, plugins_module)?;
    m.add_submodule(plugins_module)?;

    // Add serialization module
    serialization::register_serialization_module(py, m)?;

    Ok(())
}
