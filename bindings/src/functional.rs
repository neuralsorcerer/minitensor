// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::tensor::PyTensor;
use pyo3::Py;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyTuple};

fn borrow_tensor<'py>(value: &'py Bound<'py, PyAny>) -> PyResult<PyRef<'py, PyTensor>> {
    if let Ok(tensor) = value.extract::<PyRef<PyTensor>>() {
        return Ok(tensor);
    }

    let py = value.py();
    let inner = value
        .getattr(intern!(py, "_tensor"))
        .map_err(|_| PyTypeError::new_err("expected a minitensor Tensor"))?;
    Ok(inner.extract::<PyRef<PyTensor>>()?)
}

fn borrow_optional_tensor<'py>(
    value: Option<&'py Bound<'py, PyAny>>,
) -> PyResult<Option<PyRef<'py, PyTensor>>> {
    match value {
        None => Ok(None),
        Some(v) => borrow_tensor(v).map(Some),
    }
}

fn parse_normalized_shape(arg: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(value) = arg.extract::<usize>() {
        return Ok(vec![value]);
    }

    if let Ok(seq) = arg.extract::<Vec<usize>>() {
        if seq.is_empty() {
            return Err(PyValueError::new_err(
                "layer_norm requires normalized_shape to contain at least one dimension",
            ));
        }
        return Ok(seq);
    }

    Err(PyTypeError::new_err(
        "normalized_shape must be an int or sequence of ints",
    ))
}

fn to_pylist<'py>(value: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    if let Ok(list) = value.cast::<PyList>() {
        return Ok(list.clone());
    }

    let seq = value.extract::<Vec<isize>>()?;
    let list = PyList::new(value.py(), seq)?;
    Ok(list)
}

#[pyfunction]
#[pyo3(signature = (input, start_dim=None, end_dim=None))]
pub fn flatten(
    input: &Bound<PyAny>,
    start_dim: Option<isize>,
    end_dim: Option<isize>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let start = start_dim.unwrap_or(0);
    let end = end_dim.unwrap_or(-1);
    tensor.flatten(start, end)
}

#[pyfunction]
pub fn ravel(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.ravel()
}

#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn reshape(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.reshape(shape)
}

#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn view(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    reshape(input, shape)
}

#[pyfunction]
#[pyo3(signature = (input, dim, start, length))]
pub fn narrow(input: &Bound<PyAny>, dim: isize, start: usize, length: usize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.narrow(dim, start, length)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn squeeze(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.squeeze(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn unsqueeze(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.unsqueeze(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim0=0, dim1=1))]
pub fn transpose(input: &Bound<PyAny>, dim0: isize, dim1: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.transpose(Some(dim0), Some(dim1))
}

#[pyfunction]
#[pyo3(signature = (input, axis0, axis1))]
pub fn swapaxes(input: &Bound<PyAny>, axis0: isize, axis1: isize) -> PyResult<PyTensor> {
    transpose(input, axis0, axis1)
}

#[pyfunction]
#[pyo3(signature = (input, axis0, axis1))]
pub fn swapdims(input: &Bound<PyAny>, axis0: isize, axis1: isize) -> PyResult<PyTensor> {
    swapaxes(input, axis0, axis1)
}

#[pyfunction]
#[pyo3(signature = (input, *dims))]
pub fn permute(input: &Bound<PyAny>, dims: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.permute(dims)
}

#[pyfunction]
#[pyo3(signature = (input, source, destination))]
pub fn movedim(
    input: &Bound<PyAny>,
    source: &Bound<PyAny>,
    destination: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.movedim(source, destination)
}

#[pyfunction]
#[pyo3(signature = (input, source, destination))]
pub fn moveaxis(
    input: &Bound<PyAny>,
    source: &Bound<PyAny>,
    destination: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    movedim(input, source, destination)
}

#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn expand(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.expand(shape)
}

#[pyfunction]
#[pyo3(signature = (input, *repeats))]
pub fn repeat(input: &Bound<PyAny>, repeats: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.repeat(repeats)
}

#[pyfunction]
#[pyo3(signature = (input, repeats, dim=None, output_size=None))]
pub fn repeat_interleave(
    input: &Bound<PyAny>,
    repeats: &Bound<PyAny>,
    dim: Option<isize>,
    output_size: Option<usize>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.repeat_interleave(repeats, dim, output_size)
}

#[pyfunction]
#[pyo3(signature = (input, dims))]
pub fn flip(input: &Bound<PyAny>, dims: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.flip(dims)
}

#[pyfunction]
#[pyo3(signature = (input, shifts, dims=None))]
pub fn roll(
    input: &Bound<PyAny>,
    shifts: &Bound<PyAny>,
    dims: Option<&Bound<PyAny>>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.roll(shifts, dims)
}

#[pyfunction]
#[pyo3(signature = (input, chunks, dim=0))]
pub fn chunk(input: &Bound<PyAny>, chunks: usize, dim: isize) -> PyResult<Vec<PyTensor>> {
    let tensor = borrow_tensor(input)?;
    tensor.chunk(chunks, dim)
}

#[pyfunction]
#[pyo3(signature = (input, split_size_or_sections, dim=0))]
pub fn split(
    input: &Bound<PyAny>,
    split_size_or_sections: &Bound<PyAny>,
    dim: isize,
) -> PyResult<Vec<PyTensor>> {
    let tensor = borrow_tensor(input)?;
    tensor.split(split_size_or_sections, Some(dim))
}

#[pyfunction]
#[pyo3(signature = (input, dim, indices))]
pub fn index_select(
    input: &Bound<PyAny>,
    dim: isize,
    indices: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let list = to_pylist(indices)?;
    tensor.index_select(dim, &list)
}

#[pyfunction]
#[pyo3(signature = (input, dim, index))]
pub fn gather(input: &Bound<PyAny>, dim: isize, index: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let index_tensor = borrow_tensor(index)?;
    tensor.gather(dim, &*index_tensor)
}

#[pyfunction(name = "where")]
#[pyo3(signature = (condition, input, other))]
pub fn where_function(
    condition: &Bound<PyAny>,
    input: &Bound<PyAny>,
    other: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    match borrow_tensor(input) {
        Ok(tensor) => tensor.where_method(condition, other),
        Err(_) => {
            let tensor = PyTensor::from_python_value(input)?;
            tensor.where_method(condition, other)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (input, mask, value))]
pub fn masked_fill(
    input: &Bound<PyAny>,
    mask: &Bound<PyAny>,
    value: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.masked_fill(mask, value)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn softmax(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.softmax(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn log_softmax(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.log_softmax(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn logsumexp(
    input: &Bound<PyAny>,
    dim: Option<&Bound<PyAny>>,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.logsumexp(dim, Some(keepdim))
}

#[pyfunction]
pub fn relu(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.relu()
}

#[pyfunction]
#[pyo3(signature = (input, lambd=0.5))]
pub fn hardshrink(input: &Bound<PyAny>, lambd: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.hardshrink(Some(lambd))
}

#[pyfunction]
pub fn sigmoid(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sigmoid()
}

#[pyfunction]
#[pyo3(signature = (input, beta=1.0, threshold=20.0))]
pub fn softplus(input: &Bound<PyAny>, beta: f64, threshold: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.softplus(Some(beta), Some(threshold))
}

#[pyfunction]
#[pyo3(signature = (input, approximate="none"))]
pub fn gelu(input: &Bound<PyAny>, approximate: &str) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.gelu(Some(approximate))
}

#[pyfunction]
#[pyo3(signature = (input, alpha=1.0))]
pub fn elu(input: &Bound<PyAny>, alpha: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.elu(Some(alpha))
}

#[pyfunction]
pub fn selu(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.selu()
}

#[pyfunction]
pub fn silu(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.silu()
}

#[pyfunction]
pub fn softsign(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.softsign()
}

#[pyfunction]
pub fn tanh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.tanh()
}

#[pyfunction]
pub fn log1p(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.log1p()
}

#[pyfunction]
pub fn expm1(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.expm1()
}

#[pyfunction]
pub fn sin(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sin()
}

#[pyfunction]
pub fn cos(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cos()
}

#[pyfunction]
pub fn tan(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.tan()
}

#[pyfunction]
pub fn rsqrt(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.rsqrt()
}

#[pyfunction]
pub fn logaddexp(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.logaddexp(other)
}

#[pyfunction]
#[pyo3(signature = (input, diagonal=0))]
pub fn triu(input: &Bound<PyAny>, diagonal: i64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.triu(diagonal)
}

#[pyfunction]
#[pyo3(signature = (input, diagonal=0))]
pub fn tril(input: &Bound<PyAny>, diagonal: i64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.tril(diagonal)
}

#[pyfunction]
#[pyo3(signature = (input, offset=0, dim1=-2, dim2=-1))]
pub fn diagonal(
    input: &Bound<PyAny>,
    offset: isize,
    dim1: isize,
    dim2: isize,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.diagonal(offset, dim1, dim2)
}

#[pyfunction]
#[pyo3(signature = (input, offset=0, dim1=-2, dim2=-1))]
pub fn trace(input: &Bound<PyAny>, offset: isize, dim1: isize, dim2: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.trace(offset, dim1, dim2)
}

#[pyfunction]
#[pyo3(signature = (input, k, dim=None, largest=true, sorted=true))]
pub fn topk(
    input: &Bound<PyAny>,
    k: isize,
    dim: Option<isize>,
    largest: bool,
    sorted: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    if k < 0 {
        return Err(PyRuntimeError::new_err("k must be non-negative"));
    }
    let tensor = borrow_tensor(input)?;
    tensor.topk(k as usize, dim, Some(largest), Some(sorted))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, descending=false, stable=false))]
pub fn sort(
    input: &Bound<PyAny>,
    dim: Option<isize>,
    descending: bool,
    stable: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    let tensor = borrow_tensor(input)?;
    tensor.sort(dim, Some(descending), Some(stable))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, descending=false, stable=false))]
pub fn argsort(
    input: &Bound<PyAny>,
    dim: Option<isize>,
    descending: bool,
    stable: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.argsort(dim, Some(descending), Some(stable))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn median(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<Py<PyAny>> {
    let tensor = borrow_tensor(input)?;
    let (values, indices_opt) = tensor.median_with_indices(dim, keepdim)?;
    let py = input.py();
    if dim.is_some() {
        let indices = indices_opt.ok_or_else(|| {
            PyRuntimeError::new_err("median returned no indices for the requested dimension")
        })?;
        let values_any: Py<PyAny> = Py::new(py, values)?.into();
        let indices_any: Py<PyAny> = Py::new(py, indices)?.into();
        let tuple = PyTuple::new(py, [values_any, indices_any])?;
        let tuple_py: Py<PyTuple> = tuple.into();
        Ok(tuple_py.into())
    } else {
        let values_py: Py<PyTensor> = Py::new(py, values)?;
        Ok(values_py.into())
    }
}

#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=1e-5))]
pub fn layer_norm(
    input: &Bound<PyAny>,
    normalized_shape: &Bound<PyAny>,
    weight: Option<&Bound<PyAny>>,
    bias: Option<&Bound<PyAny>>,
    eps: f64,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let shape = parse_normalized_shape(normalized_shape)?;
    let weight_tensor = borrow_optional_tensor(weight)?;
    let bias_tensor = borrow_optional_tensor(bias)?;
    tensor.layer_norm(
        shape,
        weight_tensor.as_deref(),
        bias_tensor.as_deref(),
        Some(eps),
    )
}

#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(dim))
}

#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn stack(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
    PyTensor::stack(tensors, Some(dim))
}

#[pyfunction]
pub fn dot(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.dot(other)
}

pub fn register_functional_module(_py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(flatten, parent)?)?;
    parent.add_function(wrap_pyfunction!(ravel, parent)?)?;
    parent.add_function(wrap_pyfunction!(reshape, parent)?)?;
    parent.add_function(wrap_pyfunction!(view, parent)?)?;
    parent.add_function(wrap_pyfunction!(narrow, parent)?)?;
    parent.add_function(wrap_pyfunction!(squeeze, parent)?)?;
    parent.add_function(wrap_pyfunction!(unsqueeze, parent)?)?;
    parent.add_function(wrap_pyfunction!(transpose, parent)?)?;
    parent.add_function(wrap_pyfunction!(swapaxes, parent)?)?;
    parent.add_function(wrap_pyfunction!(swapdims, parent)?)?;
    parent.add_function(wrap_pyfunction!(permute, parent)?)?;
    parent.add_function(wrap_pyfunction!(movedim, parent)?)?;
    parent.add_function(wrap_pyfunction!(moveaxis, parent)?)?;
    parent.add_function(wrap_pyfunction!(expand, parent)?)?;
    parent.add_function(wrap_pyfunction!(repeat, parent)?)?;
    parent.add_function(wrap_pyfunction!(repeat_interleave, parent)?)?;
    parent.add_function(wrap_pyfunction!(flip, parent)?)?;
    parent.add_function(wrap_pyfunction!(roll, parent)?)?;
    parent.add_function(wrap_pyfunction!(chunk, parent)?)?;
    parent.add_function(wrap_pyfunction!(split, parent)?)?;
    parent.add_function(wrap_pyfunction!(index_select, parent)?)?;
    parent.add_function(wrap_pyfunction!(gather, parent)?)?;
    parent.add_function(wrap_pyfunction!(where_function, parent)?)?;
    parent.add_function(wrap_pyfunction!(masked_fill, parent)?)?;
    parent.add_function(wrap_pyfunction!(softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(log_softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(logsumexp, parent)?)?;
    parent.add_function(wrap_pyfunction!(relu, parent)?)?;
    parent.add_function(wrap_pyfunction!(hardshrink, parent)?)?;
    parent.add_function(wrap_pyfunction!(sigmoid, parent)?)?;
    parent.add_function(wrap_pyfunction!(softplus, parent)?)?;
    parent.add_function(wrap_pyfunction!(gelu, parent)?)?;
    parent.add_function(wrap_pyfunction!(elu, parent)?)?;
    parent.add_function(wrap_pyfunction!(selu, parent)?)?;
    parent.add_function(wrap_pyfunction!(silu, parent)?)?;
    parent.add_function(wrap_pyfunction!(softsign, parent)?)?;
    parent.add_function(wrap_pyfunction!(tanh, parent)?)?;
    parent.add_function(wrap_pyfunction!(log1p, parent)?)?;
    parent.add_function(wrap_pyfunction!(expm1, parent)?)?;
    parent.add_function(wrap_pyfunction!(sin, parent)?)?;
    parent.add_function(wrap_pyfunction!(cos, parent)?)?;
    parent.add_function(wrap_pyfunction!(tan, parent)?)?;
    parent.add_function(wrap_pyfunction!(rsqrt, parent)?)?;
    parent.add_function(wrap_pyfunction!(logaddexp, parent)?)?;
    parent.add_function(wrap_pyfunction!(triu, parent)?)?;
    parent.add_function(wrap_pyfunction!(tril, parent)?)?;
    parent.add_function(wrap_pyfunction!(diagonal, parent)?)?;
    parent.add_function(wrap_pyfunction!(trace, parent)?)?;
    parent.add_function(wrap_pyfunction!(topk, parent)?)?;
    parent.add_function(wrap_pyfunction!(sort, parent)?)?;
    parent.add_function(wrap_pyfunction!(argsort, parent)?)?;
    parent.add_function(wrap_pyfunction!(median, parent)?)?;
    parent.add_function(wrap_pyfunction!(layer_norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(cat, parent)?)?;
    parent.add_function(wrap_pyfunction!(stack, parent)?)?;
    parent.add_function(wrap_pyfunction!(dot, parent)?)?;
    Ok(())
}
