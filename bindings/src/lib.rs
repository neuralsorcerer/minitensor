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
    m.add_function(wrap_pyfunction!(manual_seed, m)?)?;

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

#[pyfunction]
fn manual_seed(seed: u64) -> PyResult<()> {
    engine::manual_seed(seed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::tensor::Shape;
    use engine::{DataType, Device, Tensor};

    #[test]
    fn core_module_registers_expected_symbols() {
        Python::attach(|py| -> PyResult<()> {
            let module = PyModule::new(py, "_core_test")?;
            _core(py, &module)?;

            assert!(module.getattr("__version__").is_ok());
            assert!(module.getattr("Tensor").is_ok());
            assert!(module.getattr("Shape").is_ok());
            assert!(module.getattr("Device").is_ok());
            assert!(module.getattr("functional").is_ok());
            assert!(module.getattr("debug").is_ok());
            assert!(module.getattr("numpy_compat").is_ok());
            assert!(module.getattr("plugins").is_ok());
            assert!(module.getattr("nn").is_ok());
            assert!(module.getattr("optim").is_ok());

            for function in [
                "get_gradient",
                "clear_autograd_graph",
                "is_autograd_graph_consumed",
                "mark_autograd_graph_consumed",
                "get_default_dtype",
                "set_default_dtype",
                "manual_seed",
            ] {
                assert!(module.getattr(function).is_ok(), "missing {function}");
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn autograd_graph_helpers_round_trip() {
        clear_autograd_graph().unwrap();
        assert!(!is_autograd_graph_consumed().unwrap());
        mark_autograd_graph_consumed().unwrap();
        assert!(is_autograd_graph_consumed().unwrap());
        clear_autograd_graph().unwrap();
        assert!(!is_autograd_graph_consumed().unwrap());
    }

    #[test]
    fn get_gradient_returns_none_when_no_gradient_available() {
        let tensor = Tensor::zeros(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let py_tensor = PyTensor::from_tensor(tensor);

        assert!(get_gradient(&py_tensor).unwrap().is_none());
    }

    #[test]
    fn dtype_helpers_and_manual_seed_pyfunctions_work() {
        set_default_dtype("float32").unwrap();
        assert_eq!(get_default_dtype().unwrap(), "float32");

        set_default_dtype("float64").unwrap();
        assert_eq!(get_default_dtype().unwrap(), "float64");

        manual_seed(12345).unwrap();
    }
}
