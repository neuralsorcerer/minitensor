// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::tensor::PyTensor;
use crate::error::_convert_error;
use engine::TensorIndex;
use engine::operations::arithmetic::{mul, sub};
use engine::operations::shape_ops::concatenate as tensor_concatenate;

/// NumPy-style array creation functions
#[pymodule]
pub fn numpy_compat(_py: Python, m: &PyModule) -> PyResult<()> {
    // Array creation functions
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    
    // Array manipulation functions
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(hsplit, m)?)?;
    m.add_function(wrap_pyfunction!(vsplit, m)?)?;
    
    // Mathematical functions
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    
    // Comparison functions
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;
    
    // Statistical functions
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_std, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    
    Ok(())
}

/// Create a tensor of zeros with the same shape and dtype as input
#[pyfunction]
fn zeros_like(tensor: &PyTensor, dtype: Option<&str>) -> PyResult<PyTensor> {
    let shape = tensor.shape();
    let tensor_dtype = tensor.dtype();
    let dtype_str = dtype.unwrap_or(&tensor_dtype);
    PyTensor::zeros(shape, Some(dtype_str), None, Some(false))
}

/// Create a tensor of ones with the same shape and dtype as input
#[pyfunction]
fn ones_like(tensor: &PyTensor, dtype: Option<&str>) -> PyResult<PyTensor> {
    let shape = tensor.shape();
    let tensor_dtype = tensor.dtype();
    let dtype_str = dtype.unwrap_or(&tensor_dtype);
    PyTensor::ones(shape, Some(dtype_str), None, Some(false))
}

/// Create an uninitialized tensor with the same shape and dtype as input
#[pyfunction]
fn empty_like(tensor: &PyTensor, dtype: Option<&str>) -> PyResult<PyTensor> {
    // For now, create zeros (proper empty would require uninitialized memory)
    zeros_like(tensor, dtype)
}

/// Create a tensor filled with a value, same shape and dtype as input
#[pyfunction]
fn full_like(tensor: &PyTensor, fill_value: f64, dtype: Option<&str>) -> PyResult<PyTensor> {
    let shape = tensor.shape();
    let tensor_dtype = tensor.dtype();
    let dtype_str = dtype.unwrap_or(&tensor_dtype);
    PyTensor::full(shape, fill_value, Some(dtype_str), None, Some(false))
}

/// Concatenate tensors along an axis
#[pyfunction]
fn concatenate(tensors: &PyList, axis: Option<usize>) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, axis)
}

/// Stack tensors along a new axis
#[pyfunction]
fn stack(tensors: &PyList, axis: Option<usize>) -> PyResult<PyTensor> {
    PyTensor::stack(tensors, axis)
}

/// Stack tensors vertically (row-wise)
#[pyfunction]
fn vstack(tensors: &PyList) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(0))
}

/// Stack tensors horizontally (column-wise)
#[pyfunction]
fn hstack(tensors: &PyList) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(1))
}

/// Split tensor into multiple sub-tensors
#[pyfunction]
fn split(tensor: &PyTensor, sections: usize, axis: Option<usize>) -> PyResult<Vec<PyTensor>> {
    tensor.split(sections, axis)
}

/// Split tensor horizontally
#[pyfunction]
fn hsplit(tensor: &PyTensor, sections: usize) -> PyResult<Vec<PyTensor>> {
    tensor.split(sections, Some(1))
}

/// Split tensor vertically
#[pyfunction]
fn vsplit(tensor: &PyTensor, sections: usize) -> PyResult<Vec<PyTensor>> {
    tensor.split(sections, Some(0))
}

/// Dot product of two tensors
#[pyfunction]
fn dot(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    // For 1D tensors, compute inner product
    // For 2D tensors, use matrix multiplication
    if a.ndim() == 1 && b.ndim() == 1 {
        // Inner product for 1D tensors
        let product = a.__mul__(b)?;
        product.sum(None, Some(false))
    } else {
        // Matrix multiplication for higher dimensions
        a.matmul(b)
    }
}

/// Matrix multiplication
#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    a.matmul(b)
}

/// Cross product of two tensors
#[pyfunction]
fn cross(a: &PyTensor, b: &PyTensor, _axis: Option<i32>) -> PyResult<PyTensor> {
    if a.shape().len() != 1 || b.shape().len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Cross product currently only supports 1D tensors",
        ));
    }
    if a.shape()[0] != 3 || b.shape()[0] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cross product requires 3D vectors",
        ));
    }
    if a.tensor().dtype() != b.tensor().dtype() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cross product requires tensors of the same dtype",
        ));
    }
    if a.tensor().device() != b.tensor().device() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cross product requires tensors on the same device",
        ));
    }

    let a0 = a.tensor().index(&[TensorIndex::Index(0)]).map_err(_convert_error)?;
    let a1 = a.tensor().index(&[TensorIndex::Index(1)]).map_err(_convert_error)?;
    let a2 = a.tensor().index(&[TensorIndex::Index(2)]).map_err(_convert_error)?;
    let b0 = b.tensor().index(&[TensorIndex::Index(0)]).map_err(_convert_error)?;
    let b1 = b.tensor().index(&[TensorIndex::Index(1)]).map_err(_convert_error)?;
    let b2 = b.tensor().index(&[TensorIndex::Index(2)]).map_err(_convert_error)?;

    let c0 = sub(&mul(&a1, &b2).map_err(_convert_error)?,
                 &mul(&a2, &b1).map_err(_convert_error)?)
        .map_err(_convert_error)?
        .unsqueeze(0)
        .map_err(_convert_error)?;
    let c1 = sub(&mul(&a2, &b0).map_err(_convert_error)?,
                 &mul(&a0, &b2).map_err(_convert_error)?)
        .map_err(_convert_error)?
        .unsqueeze(0)
        .map_err(_convert_error)?;
    let c2 = sub(&mul(&a0, &b1).map_err(_convert_error)?,
                 &mul(&a1, &b0).map_err(_convert_error)?)
        .map_err(_convert_error)?
        .unsqueeze(0)
        .map_err(_convert_error)?;

    let result = tensor_concatenate(&[&c0, &c1, &c2], 0).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Check if arrays are approximately equal
#[pyfunction]
fn allclose(a: &PyTensor, b: &PyTensor, rtol: Option<f64>, atol: Option<f64>) -> PyResult<bool> {
    a.allclose(b, rtol, atol)
}

/// Check if arrays are exactly equal
#[pyfunction]
fn array_equal(a: &PyTensor, b: &PyTensor) -> PyResult<bool> {
    a.array_equal(b)
}

/// Compute mean along axis
#[pyfunction]
fn mean(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.mean(axis.map(|a| vec![a]), keepdims)
}

/// Compute standard deviation along axis
#[pyfunction]
fn tensor_std(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.std(axis, keepdims)
}

/// Compute variance along axis
#[pyfunction]
fn var(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.var(axis, keepdims)
}

/// Compute sum along axis
#[pyfunction]
fn sum(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.sum(axis.map(|a| vec![a]), keepdims)
}

/// Compute maximum along axis
#[pyfunction]
fn max(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.max(axis, keepdims)
}

/// Compute minimum along axis
#[pyfunction]
fn min(tensor: &PyTensor, axis: Option<usize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    tensor.min(axis, keepdims)
}