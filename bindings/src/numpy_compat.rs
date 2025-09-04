// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::operations::arithmetic::{mul, sub};
use engine::operations::shape_ops::concatenate as tensor_concatenate;
use engine::tensor::shape::Shape;
use engine::TensorIndex;
use pyo3::prelude::*;
use pyo3::types::PyList;

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

/// Cross product of two tensors along a given axis
#[pyfunction]
#[pyo3(signature = (a, b, axis=None))]
fn cross(a: &PyTensor, b: &PyTensor, axis: Option<i32>) -> PyResult<PyTensor> {
    // Determine axes for each tensor separately (allow different ranks)
    let shape_a = a.shape();
    let shape_b = b.shape();
    let ndim_a = shape_a.len();
    let ndim_b = shape_b.len();
    let mut axis_i32 = axis.unwrap_or(-1);

    let mut axis_a = axis_i32;
    if axis_a < 0 {
        axis_a += ndim_a as i32;
    }
    if axis_a < 0 || axis_a as usize >= ndim_a {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid axis for cross product",
        ));
    }

    let mut axis_b = axis_i32;
    if axis_b < 0 {
        axis_b += ndim_b as i32;
    }
    if axis_b < 0 || axis_b as usize >= ndim_b {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid axis for cross product",
        ));
    }

    let axis_a = axis_a as usize;
    let axis_b = axis_b as usize;

    if shape_a[axis_a] != 3 || shape_b[axis_b] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cross product requires dimension of size 3 along the specified axis",
        ));
    }

    // Ensure shapes are broadcastable (excluding dtype/device checks)
    let shape_a_obj: Shape = shape_a.clone().into();
    let shape_b_obj: Shape = shape_b.clone().into();
    let broadcasted_shape = shape_a_obj
        .broadcast_with(&shape_b_obj)
        .map_err(_convert_error)?;

    // Determine axis position in broadcasted result for concatenation
    let broadcast_ndim = broadcasted_shape.ndim();
    if axis_i32 < 0 {
        axis_i32 += broadcast_ndim as i32;
    }
    if axis_i32 < 0 || axis_i32 as usize >= broadcast_ndim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid axis for cross product",
        ));
    }
    let axis_out = axis_i32 as usize;

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

    // Helper to extract a component along a given axis
    let extract = |t: &PyTensor, axis: usize, idx: usize| -> PyResult<engine::tensor::Tensor> {
        let mut indices = Vec::with_capacity(t.shape().len());
        for (dim, &size) in t.shape().iter().enumerate() {
            if dim == axis {
                indices.push(TensorIndex::Index(idx));
            } else {
                indices.push(TensorIndex::Slice {
                    start: 0,
                    end: size,
                    step: 1,
                });
            }
        }
        t.tensor().index(&indices).map_err(_convert_error)
    };

    let a0 = extract(a, axis_a, 0)?;
    let a1 = extract(a, axis_a, 1)?;
    let a2 = extract(a, axis_a, 2)?;
    let b0 = extract(b, axis_b, 0)?;
    let b1 = extract(b, axis_b, 1)?;
    let b2 = extract(b, axis_b, 2)?;

    let c0 = sub(
        &mul(&a1, &b2).map_err(_convert_error)?,
        &mul(&a2, &b1).map_err(_convert_error)?,
    )
    .map_err(_convert_error)?
    .unsqueeze(axis_out)
    .map_err(_convert_error)?;
    let c1 = sub(
        &mul(&a2, &b0).map_err(_convert_error)?,
        &mul(&a0, &b2).map_err(_convert_error)?,
    )
    .map_err(_convert_error)?
    .unsqueeze(axis_out)
    .map_err(_convert_error)?;
    let c2 = sub(
        &mul(&a0, &b1).map_err(_convert_error)?,
        &mul(&a1, &b0).map_err(_convert_error)?,
    )
    .map_err(_convert_error)?
    .unsqueeze(axis_out)
    .map_err(_convert_error)?;

    let result = tensor_concatenate(&[&c0, &c1, &c2], axis_out).map_err(_convert_error)?;
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
fn tensor_std(
    tensor: &PyTensor,
    axis: Option<usize>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
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
