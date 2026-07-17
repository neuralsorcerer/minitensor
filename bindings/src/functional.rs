// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::tensor::{Shape, TensorData};
use engine::{DataType, Device, Tensor};
use pyo3::Py;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyTuple};
use std::sync::Arc;

pub(crate) fn borrow_tensor<'py>(value: &'py Bound<'py, PyAny>) -> PyResult<PyRef<'py, PyTensor>> {
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

fn one_hot_input_to_tensor(input: &Bound<PyAny>) -> PyResult<Tensor> {
    if let Ok(tensor) = borrow_tensor(input) {
        return Ok(tensor.tensor().clone());
    }

    let input_dtype = PyTensor::infer_python_dtype(input).unwrap_or(DataType::Int64);
    Ok(PyTensor::from_python_value_with_dtype(input, input_dtype)?
        .tensor()
        .clone())
}

fn one_hot_labels(tensor: &Tensor) -> PyResult<Vec<i64>> {
    if !tensor.device().is_cpu() {
        return Err(PyValueError::new_err(
            "one_hot currently requires input labels on the CPU",
        ));
    }

    match tensor.dtype() {
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .map(|slice| slice.to_vec())
            .ok_or_else(|| PyValueError::new_err("one_hot could not read int64 labels")),
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .map(|slice| slice.iter().map(|&value| i64::from(value)).collect())
            .ok_or_else(|| PyValueError::new_err("one_hot could not read int32 labels")),
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .map(|slice| slice.iter().map(|&value| i64::from(value)).collect())
            .ok_or_else(|| PyValueError::new_err("one_hot could not read bool labels")),
        dtype => Err(PyTypeError::new_err(format!(
            "one_hot input must have an integer or bool dtype, got {dtype:?}",
        ))),
    }
}

fn fill_one_hot<T: Copy>(
    data: &mut [T],
    labels: &[i64],
    num_classes: usize,
    one: T,
) -> PyResult<()> {
    for (row, &class_id) in labels.iter().enumerate() {
        if class_id as usize >= num_classes {
            return Err(PyValueError::new_err(format!(
                "class value {class_id} is outside the valid range [0, {num_classes})",
            )));
        }
        data[row * num_classes + class_id as usize] = one;
    }
    Ok(())
}

fn make_one_hot_data(
    labels: &[i64],
    num_classes: usize,
    dtype: DataType,
    device: Device,
) -> PyResult<Arc<TensorData>> {
    let output_len = labels
        .len()
        .checked_mul(num_classes)
        .ok_or_else(|| PyValueError::new_err("one_hot output size overflow"))?;

    macro_rules! build_data {
        ($ty:ty, $zero:expr, $one:expr, $ctor:ident) => {{
            let mut data = vec![$zero; output_len];
            fill_one_hot::<$ty>(&mut data, labels, num_classes, $one)?;
            Ok(Arc::new(TensorData::$ctor(data, device)))
        }};
    }

    match dtype {
        DataType::Float32 => build_data!(f32, 0.0_f32, 1.0_f32, from_vec_f32),
        DataType::Float64 => build_data!(f64, 0.0_f64, 1.0_f64, from_vec_f64),
        DataType::Int32 => build_data!(i32, 0_i32, 1_i32, from_vec_i32),
        DataType::Int64 => build_data!(i64, 0_i64, 1_i64, from_vec_i64),
        DataType::Bool => build_data!(bool, false, true, from_vec_bool),
    }
}

fn bincount_labels(tensor: &Tensor) -> PyResult<Vec<usize>> {
    if !tensor.device().is_cpu() {
        return Err(PyValueError::new_err(
            "bincount currently requires input labels on the CPU",
        ));
    }

    let values: Vec<i64> = match tensor.dtype() {
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .map(|slice| slice.to_vec())
            .ok_or_else(|| PyValueError::new_err("bincount could not read int64 input"))?,
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .map(|slice| slice.iter().map(|&value| i64::from(value)).collect())
            .ok_or_else(|| PyValueError::new_err("bincount could not read int32 input"))?,
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .map(|slice| slice.iter().map(|&value| i64::from(value)).collect())
            .ok_or_else(|| PyValueError::new_err("bincount could not read bool input"))?,
        dtype => {
            return Err(PyTypeError::new_err(format!(
                "bincount input must have an integer or bool dtype, got {dtype:?}",
            )));
        }
    };

    values
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "bincount input values must be non-negative, got {value}",
                ))
            })
        })
        .collect()
}

fn bincount_output_len(labels: &[usize], minlength: isize) -> PyResult<usize> {
    if minlength < 0 {
        return Err(PyValueError::new_err("minlength must be non-negative"));
    }

    let inferred = labels
        .iter()
        .copied()
        .max()
        .map(|value| {
            value
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("bincount output size overflow"))
        })
        .transpose()?
        .unwrap_or(0);
    Ok(inferred.max(minlength as usize))
}

fn bincount_tensor(data: TensorData, dtype: DataType, output_len: usize) -> Tensor {
    Tensor::new(
        Arc::new(data),
        Shape::new(vec![output_len]),
        dtype,
        Device::cpu(),
        false,
    )
}

fn make_bincount_tensor(
    labels: &[usize],
    weights: Option<&Tensor>,
    minlength: isize,
) -> PyResult<Tensor> {
    let output_len = bincount_output_len(labels, minlength)?;

    match weights {
        None => {
            let mut counts = vec![0_i64; output_len];
            for &label in labels {
                counts[label] = counts[label].checked_add(1).ok_or_else(|| {
                    PyValueError::new_err("bincount count overflow for int64 output")
                })?;
            }
            Ok(bincount_tensor(
                TensorData::from_vec_i64(counts, Device::cpu()),
                DataType::Int64,
                output_len,
            ))
        }
        Some(weight_tensor) => {
            if !weight_tensor.device().is_cpu() {
                return Err(PyValueError::new_err(
                    "bincount currently requires weights on the CPU",
                ));
            }
            match weight_tensor.dtype() {
                DataType::Float32 => {
                    let values = weight_tensor.data().as_f32_slice().ok_or_else(|| {
                        PyValueError::new_err("bincount could not read float32 weights")
                    })?;
                    let mut counts = vec![0.0_f32; output_len];
                    for (&label, &weight) in labels.iter().zip(values) {
                        counts[label] += weight;
                    }
                    Ok(bincount_tensor(
                        TensorData::from_vec_f32(counts, Device::cpu()),
                        DataType::Float32,
                        output_len,
                    ))
                }
                DataType::Float64 => {
                    let values = weight_tensor.data().as_f64_slice().ok_or_else(|| {
                        PyValueError::new_err("bincount could not read float64 weights")
                    })?;
                    let mut counts = vec![0.0_f64; output_len];
                    for (&label, &weight) in labels.iter().zip(values) {
                        counts[label] += weight;
                    }
                    Ok(bincount_tensor(
                        TensorData::from_vec_f64(counts, Device::cpu()),
                        DataType::Float64,
                        output_len,
                    ))
                }
                dtype => Err(PyTypeError::new_err(format!(
                    "bincount weights must have a floating-point dtype, got {dtype:?}",
                ))),
            }
        }
    }
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
#[pyo3(signature = (input, min=None, max=None))]
pub fn clip(
    input: &Bound<PyAny>,
    min: Option<&Bound<PyAny>>,
    max: Option<&Bound<PyAny>>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clip(min, max)
}

#[pyfunction]
#[pyo3(signature = (input, min=None, max=None))]
pub fn clamp(
    input: &Bound<PyAny>,
    min: Option<&Bound<PyAny>>,
    max: Option<&Bound<PyAny>>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clamp(min, max)
}

#[pyfunction]
pub fn clamp_min(input: &Bound<PyAny>, min: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clamp_min(min)
}

#[pyfunction]
pub fn clamp_max(input: &Bound<PyAny>, max: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clamp_max(max)
}

#[pyfunction]
#[pyo3(signature = (input, decimals=0))]
pub fn round(input: &Bound<PyAny>, decimals: i32) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.round(decimals)
}

#[pyfunction]
pub fn floor(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.floor()
}

#[pyfunction]
pub fn ceil(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.ceil()
}

#[pyfunction]
pub fn sign(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sign()
}

#[pyfunction]
pub fn reciprocal(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.reciprocal()
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
    tensor.index_select(dim, indices)
}

#[pyfunction]
#[pyo3(signature = (input, dim, index))]
pub fn gather(input: &Bound<PyAny>, dim: isize, index: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let index_tensor = borrow_tensor(index)?;
    tensor.gather(dim, &index_tensor)
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
#[pyo3(signature = (input, weights=None, minlength=0))]
pub fn bincount(
    input: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    minlength: isize,
) -> PyResult<PyTensor> {
    let input_tensor = one_hot_input_to_tensor(input)?;
    if input_tensor.shape().ndim() != 1 {
        return Err(PyValueError::new_err("bincount input must be 1-D"));
    }

    let labels = bincount_labels(&input_tensor)?;
    let weights_tensor = borrow_optional_tensor(weights)?;
    if let Some(weight_tensor) = weights_tensor.as_deref()
        && weight_tensor.tensor().shape().dims() != input_tensor.shape().dims()
    {
        return Err(PyValueError::new_err(
            "weights must have the same shape as input",
        ));
    }
    let output = make_bincount_tensor(
        &labels,
        weights_tensor.as_deref().map(PyTensor::tensor),
        minlength,
    )?;
    Ok(PyTensor::from_tensor(output))
}

#[pyfunction]
#[pyo3(signature = (input, num_classes=None, dtype="float32"))]
pub fn one_hot(
    input: &Bound<PyAny>,
    num_classes: Option<isize>,
    dtype: &str,
) -> PyResult<PyTensor> {
    let input_tensor = one_hot_input_to_tensor(input)?;
    let labels = one_hot_labels(&input_tensor)?;

    let inferred_classes = labels.iter().try_fold(None::<i64>, |max_label, &label| {
        if label < 0 {
            Err(PyValueError::new_err(format!(
                "one_hot class values must be non-negative, got {label}",
            )))
        } else {
            Ok(Some(max_label.map_or(label, |current| current.max(label))))
        }
    })?;

    let classes = match num_classes {
        Some(value) if value < 0 => {
            return Err(PyValueError::new_err(
                "num_classes must be non-negative when provided",
            ));
        }
        Some(value) => value as usize,
        None => inferred_classes
            .map(|max_label| (max_label as usize) + 1)
            .ok_or_else(|| {
                PyValueError::new_err("num_classes must be provided when one_hot input is empty")
            })?,
    };

    let output_dtype = crate::dtype::parse_dtype(dtype)?;
    let mut output_shape = input_tensor.shape().dims().to_vec();
    output_shape.push(classes);
    let data = make_one_hot_data(&labels, classes, output_dtype, input_tensor.device())?;
    let output = Tensor::new(
        data,
        Shape::new(output_shape),
        output_dtype,
        input_tensor.device(),
        false,
    );
    Ok(PyTensor::from_tensor(output))
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
#[pyo3(signature = (input, mask, dim=None))]
pub fn masked_softmax(
    input: &Bound<PyAny>,
    mask: &Bound<PyAny>,
    dim: Option<isize>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.masked_softmax(mask, dim)
}

#[pyfunction]
#[pyo3(signature = (input, mask, dim=None))]
pub fn masked_log_softmax(
    input: &Bound<PyAny>,
    mask: &Bound<PyAny>,
    dim: Option<isize>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.masked_log_softmax(mask, dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn sum(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sum(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn prod(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.prod(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn mean(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.mean(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn all(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.all(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn any(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.any(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn max(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<Py<PyAny>> {
    let tensor = borrow_tensor(input)?;
    let py = input.py();
    if let Some(dim) = dim {
        let (values, indices) = tensor
            .tensor()
            .max_with_indices(dim, keepdim)
            .map_err(_convert_error)?;
        let values_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(values))?.into();
        let indices_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(indices))?.into();
        let tuple = PyTuple::new(py, [values_any, indices_any])?;
        let tuple_py: Py<PyTuple> = tuple.into();
        Ok(tuple_py.into())
    } else {
        let values: Py<PyTensor> = Py::new(py, tensor.max_values(None, keepdim)?)?;
        Ok(values.into())
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn min(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<Py<PyAny>> {
    let tensor = borrow_tensor(input)?;
    let py = input.py();
    if let Some(dim) = dim {
        let (values, indices) = tensor
            .tensor()
            .min_with_indices(dim, keepdim)
            .map_err(_convert_error)?;
        let values_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(values))?.into();
        let indices_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(indices))?.into();
        let tuple = PyTuple::new(py, [values_any, indices_any])?;
        let tuple_py: Py<PyTuple> = tuple.into();
        Ok(tuple_py.into())
    } else {
        let values: Py<PyTensor> = Py::new(py, tensor.min_values(None, keepdim)?)?;
        Ok(values.into())
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmax(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.argmax(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmin(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.argmin(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn cumsum(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cumsum(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn cumprod(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cumprod(dim)
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, unbiased=true, keepdim=false))]
#[pyo3(name = "std")]
pub fn std_fn(
    input: &Bound<PyAny>,
    dim: Option<&Bound<PyAny>>,
    unbiased: bool,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.std(dim, Some(unbiased), Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, unbiased=true, keepdim=false))]
pub fn var(
    input: &Bound<PyAny>,
    dim: Option<&Bound<PyAny>>,
    unbiased: bool,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.var(dim, Some(unbiased), Some(keepdim))
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
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nansum(
    input: &Bound<PyAny>,
    dim: Option<&Bound<PyAny>>,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nansum(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanmean(
    input: &Bound<PyAny>,
    dim: Option<&Bound<PyAny>>,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nanmean(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanmax(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<Py<PyAny>> {
    let tensor = borrow_tensor(input)?;
    let py = input.py();
    if let Some(dim) = dim {
        let (values, indices) = tensor
            .tensor()
            .nanmax_with_indices(dim, keepdim)
            .map_err(_convert_error)?;
        let values_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(values))?.into();
        let indices_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(indices))?.into();
        let tuple = PyTuple::new(py, [values_any, indices_any])?;
        let tuple_py: Py<PyTuple> = tuple.into();
        Ok(tuple_py.into())
    } else {
        let values: Py<PyTensor> = Py::new(py, tensor.nanmax_values(None, keepdim)?)?;
        Ok(values.into())
    }
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanmin(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<Py<PyAny>> {
    let tensor = borrow_tensor(input)?;
    let py = input.py();
    if let Some(dim) = dim {
        let (values, indices) = tensor
            .tensor()
            .nanmin_with_indices(dim, keepdim)
            .map_err(_convert_error)?;
        let values_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(values))?.into();
        let indices_any: Py<PyAny> = Py::new(py, PyTensor::from_tensor(indices))?.into();
        let tuple = PyTuple::new(py, [values_any, indices_any])?;
        let tuple_py: Py<PyTuple> = tuple.into();
        Ok(tuple_py.into())
    } else {
        let values: Py<PyTensor> = Py::new(py, tensor.nanmin_values(None, keepdim)?)?;
        Ok(values.into())
    }
}

fn finite_predicate(
    input: &Bound<PyAny>,
    predicate: impl FnOnce(&Tensor) -> engine::error::Result<Tensor>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    predicate(tensor.tensor())
        .map(PyTensor::from_tensor)
        .map_err(_convert_error)
}

#[pyfunction]
pub fn isnan(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isnan)
}

#[pyfunction]
pub fn isinf(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isinf)
}

#[pyfunction]
pub fn isfinite(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isfinite)
}

#[pyfunction]
#[pyo3(signature = (input, nan=0.0, posinf=None, neginf=None))]
pub fn nan_to_num(
    input: &Bound<PyAny>,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nan_to_num(nan, posinf, neginf)
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
pub fn asin(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.asin()
}

#[pyfunction]
pub fn acos(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.acos()
}

#[pyfunction]
pub fn atan(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.atan()
}

#[pyfunction]
pub fn sinh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sinh()
}

#[pyfunction]
pub fn cosh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cosh()
}

#[pyfunction]
pub fn asinh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.asinh()
}

#[pyfunction]
pub fn acosh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.acosh()
}

#[pyfunction]
pub fn atanh(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.atanh()
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
pub fn maximum(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.maximum(other)
}

#[pyfunction]
pub fn minimum(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.minimum(other)
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
pub fn solve(lhs: &Bound<PyAny>, rhs: &Bound<PyAny>) -> PyResult<PyTensor> {
    let lhs_tensor = borrow_tensor(lhs)?;
    lhs_tensor.solve(rhs)
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
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanmedian(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nanmedian(dim, Some(keepdim))
}

#[pyfunction]
#[pyo3(signature = (input, q, dim=None, keepdim=false, interpolation="linear"))]
pub fn quantile(
    input: &Bound<PyAny>,
    q: &Bound<PyAny>,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.quantile(q, dim, Some(keepdim), Some(interpolation))
}

#[pyfunction]
#[pyo3(signature = (input, q, dim=None, keepdim=false, interpolation="linear"))]
pub fn nanquantile(
    input: &Bound<PyAny>,
    q: &Bound<PyAny>,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nanquantile(q, dim, Some(keepdim), Some(interpolation))
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

#[pyfunction]
pub fn bmm(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.bmm(other)
}

#[pyfunction]
#[pyo3(signature = (input, other, rtol=None, atol=None, equal_nan=false))]
pub fn isclose(
    input: &Bound<PyAny>,
    other: &Bound<PyAny>,
    rtol: Option<f64>,
    atol: Option<f64>,
    equal_nan: bool,
) -> PyResult<PyTensor> {
    let lhs = PyTensor::from_python_value(input)?;
    lhs.isclose(other, rtol, atol, equal_nan)
}

#[pyfunction]
pub fn array_equal(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<bool> {
    let lhs = PyTensor::from_python_value(input)?;
    let rhs = PyTensor::from_python_value(other)?;
    lhs.array_equal(&rhs)
}

#[pyfunction]
#[pyo3(signature = (input, other, rtol=None, atol=None, equal_nan=false))]
pub fn allclose(
    input: &Bound<PyAny>,
    other: &Bound<PyAny>,
    rtol: Option<f64>,
    atol: Option<f64>,
    equal_nan: bool,
) -> PyResult<bool> {
    let lhs = PyTensor::from_python_value(input)?;
    let rhs = PyTensor::from_python_value(other)?;
    lhs.allclose(&rhs, rtol, atol, equal_nan)
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
    parent.add_function(wrap_pyfunction!(clip, parent)?)?;
    parent.add_function(wrap_pyfunction!(clamp, parent)?)?;
    parent.add_function(wrap_pyfunction!(clamp_min, parent)?)?;
    parent.add_function(wrap_pyfunction!(clamp_max, parent)?)?;
    parent.add_function(wrap_pyfunction!(round, parent)?)?;
    parent.add_function(wrap_pyfunction!(floor, parent)?)?;
    parent.add_function(wrap_pyfunction!(ceil, parent)?)?;
    parent.add_function(wrap_pyfunction!(sign, parent)?)?;
    parent.add_function(wrap_pyfunction!(reciprocal, parent)?)?;
    parent.add_function(wrap_pyfunction!(chunk, parent)?)?;
    parent.add_function(wrap_pyfunction!(split, parent)?)?;
    parent.add_function(wrap_pyfunction!(index_select, parent)?)?;
    parent.add_function(wrap_pyfunction!(gather, parent)?)?;
    parent.add_function(wrap_pyfunction!(where_function, parent)?)?;
    parent.add_function(wrap_pyfunction!(one_hot, parent)?)?;
    parent.add_function(wrap_pyfunction!(bincount, parent)?)?;
    parent.add_function(wrap_pyfunction!(masked_fill, parent)?)?;
    parent.add_function(wrap_pyfunction!(softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(log_softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(masked_softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(masked_log_softmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(sum, parent)?)?;
    parent.add_function(wrap_pyfunction!(prod, parent)?)?;
    parent.add_function(wrap_pyfunction!(mean, parent)?)?;
    parent.add_function(wrap_pyfunction!(all, parent)?)?;
    parent.add_function(wrap_pyfunction!(any, parent)?)?;
    parent.add_function(wrap_pyfunction!(max, parent)?)?;
    parent.add_function(wrap_pyfunction!(min, parent)?)?;
    parent.add_function(wrap_pyfunction!(argmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(argmin, parent)?)?;
    parent.add_function(wrap_pyfunction!(cumsum, parent)?)?;
    parent.add_function(wrap_pyfunction!(cumprod, parent)?)?;
    parent.add_function(wrap_pyfunction!(std_fn, parent)?)?;
    parent.add_function(wrap_pyfunction!(var, parent)?)?;
    parent.add_function(wrap_pyfunction!(logsumexp, parent)?)?;
    parent.add_function(wrap_pyfunction!(nansum, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmean, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmin, parent)?)?;
    parent.add_function(wrap_pyfunction!(isnan, parent)?)?;
    parent.add_function(wrap_pyfunction!(isinf, parent)?)?;
    parent.add_function(wrap_pyfunction!(isfinite, parent)?)?;
    parent.add_function(wrap_pyfunction!(nan_to_num, parent)?)?;
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
    parent.add_function(wrap_pyfunction!(asin, parent)?)?;
    parent.add_function(wrap_pyfunction!(acos, parent)?)?;
    parent.add_function(wrap_pyfunction!(atan, parent)?)?;
    parent.add_function(wrap_pyfunction!(sinh, parent)?)?;
    parent.add_function(wrap_pyfunction!(cosh, parent)?)?;
    parent.add_function(wrap_pyfunction!(asinh, parent)?)?;
    parent.add_function(wrap_pyfunction!(acosh, parent)?)?;
    parent.add_function(wrap_pyfunction!(atanh, parent)?)?;
    parent.add_function(wrap_pyfunction!(rsqrt, parent)?)?;
    parent.add_function(wrap_pyfunction!(logaddexp, parent)?)?;
    parent.add_function(wrap_pyfunction!(maximum, parent)?)?;
    parent.add_function(wrap_pyfunction!(minimum, parent)?)?;
    parent.add_function(wrap_pyfunction!(triu, parent)?)?;
    parent.add_function(wrap_pyfunction!(tril, parent)?)?;
    parent.add_function(wrap_pyfunction!(diagonal, parent)?)?;
    parent.add_function(wrap_pyfunction!(trace, parent)?)?;
    parent.add_function(wrap_pyfunction!(solve, parent)?)?;
    parent.add_function(wrap_pyfunction!(topk, parent)?)?;
    parent.add_function(wrap_pyfunction!(sort, parent)?)?;
    parent.add_function(wrap_pyfunction!(argsort, parent)?)?;
    parent.add_function(wrap_pyfunction!(median, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmedian, parent)?)?;
    parent.add_function(wrap_pyfunction!(quantile, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanquantile, parent)?)?;
    parent.add_function(wrap_pyfunction!(layer_norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(cat, parent)?)?;
    parent.add_function(wrap_pyfunction!(stack, parent)?)?;
    parent.add_function(wrap_pyfunction!(dot, parent)?)?;
    parent.add_function(wrap_pyfunction!(bmm, parent)?)?;
    parent.add_function(wrap_pyfunction!(isclose, parent)?)?;
    parent.add_function(wrap_pyfunction!(array_equal, parent)?)?;
    parent.add_function(wrap_pyfunction!(allclose, parent)?)?;
    Ok(())
}
