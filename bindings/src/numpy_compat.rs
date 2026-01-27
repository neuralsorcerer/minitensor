// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::Device;
use engine::TensorIndex;
use engine::operations::arithmetic::{mul, sub};
use engine::operations::reduction::sum as tensor_sum;
use engine::operations::shape_ops::concatenate as tensor_concatenate;
use engine::tensor::shape::Shape;
use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PyTuple};

struct LikeArgs {
    shape: Vec<usize>,
    dtype: String,
    requires_grad: bool,
    device: Device,
}

impl LikeArgs {
    fn from_source(source: &Bound<PyAny>, dtype_override: Option<&str>) -> PyResult<Self> {
        let tensor = PyTensor::from_python_value(source)?;

        Ok(Self {
            shape: tensor.shape_vec(),
            dtype: dtype_override
                .map(str::to_owned)
                .unwrap_or_else(|| tensor.dtype()),
            requires_grad: tensor.requires_grad(),
            device: tensor.tensor().device(),
        })
    }

    fn shape_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    fn py_device(&self) -> PyDevice {
        PyDevice::from_device(self.device)
    }
}

/// NumPy-style array creation functions
#[pymodule]
pub fn numpy_compat(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
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
    m.add_function(wrap_pyfunction!(where_py, m)?)?;

    // Comparison functions
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;

    // Statistical functions
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_std, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(nansum, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(nanmin, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (data, dtype=None, requires_grad=false))]
fn asarray(data: &Bound<PyAny>, dtype: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let mut tensor = PyTensor::from_python_value(data)?;
    if let Some(target_dtype) = dtype {
        if tensor.dtype() != target_dtype {
            tensor = tensor.astype(target_dtype)?;
        }
    }

    if tensor.requires_grad() != requires_grad {
        tensor.requires_grad_(requires_grad)?;
    }

    Ok(tensor)
}

/// Create a tensor of zeros with the same shape and dtype as input
#[pyfunction]
#[pyo3(signature = (tensor, dtype=None))]
fn zeros_like(tensor: &Bound<PyAny>, dtype: Option<&str>) -> PyResult<PyTensor> {
    let args = LikeArgs::from_source(tensor, dtype)?;
    let py = tensor.py();
    let shape_tuple = args.shape_tuple(py)?;
    let device = args.py_device();

    PyTensor::zeros(
        &shape_tuple,
        Some(args.dtype.as_str()),
        Some(&device),
        Some(args.requires_grad),
    )
}

/// Create a tensor of ones with the same shape and dtype as input
#[pyfunction]
#[pyo3(signature = (tensor, dtype=None))]
fn ones_like(tensor: &Bound<PyAny>, dtype: Option<&str>) -> PyResult<PyTensor> {
    let args = LikeArgs::from_source(tensor, dtype)?;
    let py = tensor.py();
    let shape_tuple = args.shape_tuple(py)?;
    let device = args.py_device();

    PyTensor::ones(
        &shape_tuple,
        Some(args.dtype.as_str()),
        Some(&device),
        Some(args.requires_grad),
    )
}

/// Create an uninitialized tensor with the same shape and dtype as input
#[pyfunction]
#[pyo3(signature = (tensor, dtype=None))]
fn empty_like(tensor: &Bound<PyAny>, dtype: Option<&str>) -> PyResult<PyTensor> {
    let args = LikeArgs::from_source(tensor, dtype)?;
    let py = tensor.py();
    let shape_tuple = args.shape_tuple(py)?;
    let device = args.py_device();

    PyTensor::empty(
        &shape_tuple,
        Some(args.dtype.as_str()),
        Some(&device),
        Some(args.requires_grad),
    )
}

/// Create a tensor filled with a value, same shape and dtype as input
#[pyfunction]
#[pyo3(signature = (tensor, fill_value, dtype=None))]
fn full_like(tensor: &Bound<PyAny>, fill_value: f64, dtype: Option<&str>) -> PyResult<PyTensor> {
    let args = LikeArgs::from_source(tensor, dtype)?;
    let py = tensor.py();
    let shape_tuple = args.shape_tuple(py)?;
    let device = args.py_device();

    PyTensor::full(
        shape_tuple.as_any(),
        fill_value,
        Some(args.dtype.as_str()),
        Some(&device),
        Some(args.requires_grad),
    )
}

/// Concatenate tensors along an axis
#[pyfunction]
fn concatenate(tensors: &Bound<PyList>, axis: Option<isize>) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, axis)
}

/// Stack tensors along a new axis
#[pyfunction]
fn stack(tensors: &Bound<PyList>, axis: Option<isize>) -> PyResult<PyTensor> {
    PyTensor::stack(tensors, axis)
}

/// Stack tensors vertically (row-wise)
#[pyfunction]
fn vstack(tensors: &Bound<PyList>) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(0))
}

/// Stack tensors horizontally (column-wise)
#[pyfunction]
fn hstack(tensors: &Bound<PyList>) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(1))
}

/// Split tensor into multiple sub-tensors
#[pyfunction]
fn split(tensor: &PyTensor, sections: usize, axis: Option<isize>) -> PyResult<Vec<PyTensor>> {
    let dim = axis.unwrap_or(0);
    tensor.chunk(sections, dim)
}

/// Split tensor horizontally
#[pyfunction]
fn hsplit(tensor: &PyTensor, sections: usize) -> PyResult<Vec<PyTensor>> {
    tensor.chunk(sections, 1)
}

/// Split tensor vertically
#[pyfunction]
fn vsplit(tensor: &PyTensor, sections: usize) -> PyResult<Vec<PyTensor>> {
    tensor.chunk(sections, 0)
}

/// Dot product of two tensors
#[pyfunction]
fn dot(a: &Bound<PyAny>, b: &Bound<PyAny>) -> PyResult<PyTensor> {
    let a_tensor = PyTensor::from_python_value(a)?;
    if a_tensor.ndim() == 1 {
        let b_tensor = PyTensor::from_python_value(b)?;
        if b_tensor.ndim() == 1 {
            let product = mul(a_tensor.tensor(), b_tensor.tensor()).map_err(_convert_error)?;
            let summed = tensor_sum(&product, None, false).map_err(_convert_error)?;
            return Ok(PyTensor::from_tensor(summed));
        }
        return a_tensor.matmul(b);
    }

    a_tensor.matmul(b)
}

/// Matrix multiplication
#[pyfunction]
fn matmul(a: &Bound<PyAny>, b: &Bound<PyAny>) -> PyResult<PyTensor> {
    let a_tensor = PyTensor::from_python_value(a)?;
    a_tensor.matmul(b)
}

/// Element-wise selection based on a boolean condition
#[pyfunction(name = "where")]
fn where_py(condition: &Bound<PyAny>, x: &Bound<PyAny>, y: &Bound<PyAny>) -> PyResult<PyTensor> {
    let x_tensor = PyTensor::from_python_value(x)?;
    x_tensor.where_method(condition, y)
}

pub(crate) fn cross_impl(a: &PyTensor, b: &PyTensor, axis: Option<i32>) -> PyResult<PyTensor> {
    // Determine axes for each tensor separately (allow different ranks)
    let shape_a = a.shape_vec();
    let shape_b = b.shape_vec();
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
    let axis_out = axis_i32 as isize;

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
        let dims = t.shape_vec();
        let mut indices = Vec::with_capacity(dims.len());
        for (dim, &size) in dims.iter().enumerate() {
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

/// Cross product of two tensors along a given axis
#[pyfunction]
#[pyo3(signature = (a, b, axis=None))]
fn cross(a: &Bound<PyAny>, b: &Bound<PyAny>, axis: Option<i32>) -> PyResult<PyTensor> {
    let a_tensor = PyTensor::from_python_value(a)?;
    let b_tensor = PyTensor::from_python_value(b)?;
    cross_impl(&a_tensor, &b_tensor, axis)
}

/// Check if arrays are approximately equal
#[pyfunction]
#[pyo3(signature = (a, b, rtol=None, atol=None))]
fn allclose(
    a: &Bound<PyAny>,
    b: &Bound<PyAny>,
    rtol: Option<f64>,
    atol: Option<f64>,
) -> PyResult<bool> {
    let a_tensor = PyTensor::from_python_value(a)?;
    let b_tensor = PyTensor::from_python_value(b)?;
    a_tensor.allclose(&b_tensor, rtol, atol)
}

/// Check if arrays are exactly equal
#[pyfunction]
fn array_equal(a: &Bound<PyAny>, b: &Bound<PyAny>) -> PyResult<bool> {
    let a_tensor = PyTensor::from_python_value(a)?;
    let b_tensor = PyTensor::from_python_value(b)?;
    a_tensor.array_equal(&b_tensor)
}

/// Compute mean along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn mean(
    tensor: &Bound<PyAny>,
    axis: Option<&Bound<PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.mean(axis, keepdims)
}

/// Compute NaN-aware mean along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn nanmean(
    tensor: &Bound<PyAny>,
    axis: Option<&Bound<PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.nanmean(axis, keepdims)
}

/// Compute standard deviation along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None, ddof=None))]
fn tensor_std(
    tensor: &Bound<PyAny>,
    axis: Option<isize>,
    keepdims: Option<bool>,
    ddof: Option<usize>,
) -> PyResult<PyTensor> {
    let ddof = ddof.unwrap_or(0);
    if ddof > 1 {
        return Err(PyValueError::new_err(
            "minitensor only supports ddof values of 0 or 1",
        ));
    }
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.std(axis, keepdims, Some(ddof == 1))
}

/// Compute variance along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None, ddof=None))]
fn var(
    tensor: &Bound<PyAny>,
    axis: Option<isize>,
    keepdims: Option<bool>,
    ddof: Option<usize>,
) -> PyResult<PyTensor> {
    let ddof = ddof.unwrap_or(0);
    if ddof > 1 {
        return Err(PyValueError::new_err(
            "minitensor only supports ddof values of 0 or 1",
        ));
    }
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.var(axis, keepdims, Some(ddof == 1))
}

/// Compute product along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn prod(
    tensor: &Bound<PyAny>,
    axis: Option<&Bound<PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.prod(axis, keepdims)
}

/// Compute sum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn sum(
    tensor: &Bound<PyAny>,
    axis: Option<&Bound<PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.sum(axis, keepdims)
}

/// Compute NaN-aware sum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn nansum(
    tensor: &Bound<PyAny>,
    axis: Option<&Bound<PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    tensor.nansum(axis, keepdims)
}

/// Compute maximum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn max(tensor: &Bound<PyAny>, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    let keepdim = keepdims.unwrap_or(false);
    tensor.max_values(axis, keepdim)
}

/// Compute minimum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn min(tensor: &Bound<PyAny>, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    let keepdim = keepdims.unwrap_or(false);
    tensor.min_values(axis, keepdim)
}

/// Compute NaN-aware maximum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn nanmax(
    tensor: &Bound<PyAny>,
    axis: Option<isize>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    let keepdim = keepdims.unwrap_or(false);
    tensor.nanmax_values(axis, keepdim)
}

/// Compute NaN-aware minimum along axis
#[pyfunction]
#[pyo3(signature = (tensor, axis=None, keepdims=None))]
fn nanmin(
    tensor: &Bound<PyAny>,
    axis: Option<isize>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let tensor = PyTensor::from_python_value(tensor)?;
    let keepdim = keepdims.unwrap_or(false);
    tensor.nanmin_values(axis, keepdim)
}
