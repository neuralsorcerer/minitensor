// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use pyo3::prelude::*;
use pyo3::types::PyList;
use engine::{
    custom_ops::{
        unregister_custom_op, execute_custom_op, 
        list_custom_ops, is_custom_op_registered, examples::register_example_ops
    },
};
use crate::tensor::PyTensor;
use crate::error::_convert_error;

/// Register example custom operations
#[pyfunction]
fn register_example_custom_ops() -> PyResult<()> {
    register_example_ops()
        .map_err(_convert_error)?;
    Ok(())
}

/// Unregister a custom operation
#[pyfunction]
fn unregister_custom_op_py(name: &str) -> PyResult<()> {
    unregister_custom_op(name)
        .map_err(_convert_error)?;
    Ok(())
}

/// Execute a custom operation
#[pyfunction]
fn execute_custom_op_py(name: &str, inputs: &PyList) -> PyResult<PyTensor> {
    // Convert Python list to vector of tensor references
    let mut tensor_refs = Vec::new();
    let mut tensors = Vec::new();
    
    for item in inputs.iter() {
        let py_tensor: PyTensor = item.extract()?;
        tensors.push(py_tensor.tensor().clone());
    }
    
    // Create references
    for tensor in &tensors {
        tensor_refs.push(tensor);
    }
    
    // Execute the operation
    let result = execute_custom_op(name, &tensor_refs)
        .map_err(_convert_error)?;
    
    Ok(PyTensor::from_tensor(result))
}

/// List all registered custom operations
#[pyfunction]
fn list_custom_ops_py() -> PyResult<Vec<String>> {
    list_custom_ops()
        .map_err(_convert_error)
}

/// Check if a custom operation is registered
#[pyfunction]
fn is_custom_op_registered_py(name: &str) -> PyResult<bool> {
    is_custom_op_registered(name)
        .map_err(_convert_error)
}

/// Initialize the custom operations module
pub fn init_custom_ops_module(_py: Python, parent_module: &PyModule) -> PyResult<()> {
    // Add functions to parent module
    parent_module.add_function(wrap_pyfunction!(register_example_custom_ops, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(unregister_custom_op_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(execute_custom_op_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(list_custom_ops_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(is_custom_op_registered_py, parent_module)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_custom_ops_bindings() {
        Python::with_gil(|_| {
            // Test that we can call the functions without panicking
            let result = register_example_custom_ops();
            assert!(result.is_ok());
            
            let ops = list_custom_ops_py();
            assert!(ops.is_ok());
            
            let is_registered = is_custom_op_registered_py("swish");
            assert!(is_registered.is_ok());
        });
    }
}