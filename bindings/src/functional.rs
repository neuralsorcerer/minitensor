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

/// Collapse dimensions `start_dim` through `end_dim` into one.
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

/// A tensor with the same elements in a new shape. One dimension may be -1 to be inferred.
#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn reshape(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.reshape(shape)
}

/// Alias of `reshape`.
#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn view(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    reshape(input, shape)
}

/// A slice of `length` entries along `dim`, starting at `start`.
#[pyfunction]
#[pyo3(signature = (input, dim, start, length))]
pub fn narrow(input: &Bound<PyAny>, dim: isize, start: usize, length: usize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.narrow(dim, start, length)
}

/// Drop dimensions of size 1, or just the one named by `dim`.
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn squeeze(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.squeeze(dim)
}

/// Insert a dimension of size 1 at `dim`, which may be one past the last axis.
#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn unsqueeze(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.unsqueeze(dim)
}

/// Swap two dimensions.
#[pyfunction]
#[pyo3(signature = (input, dim0=0, dim1=1))]
pub fn transpose(input: &Bound<PyAny>, dim0: isize, dim1: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.transpose(Some(dim0), Some(dim1))
}

/// Alias of `transpose`, under its array-library spelling.
#[pyfunction]
#[pyo3(signature = (input, axis0, axis1))]
pub fn swapaxes(input: &Bound<PyAny>, axis0: isize, axis1: isize) -> PyResult<PyTensor> {
    transpose(input, axis0, axis1)
}

/// Alias of `transpose`.
#[pyfunction]
#[pyo3(signature = (input, axis0, axis1))]
pub fn swapdims(input: &Bound<PyAny>, axis0: isize, axis1: isize) -> PyResult<PyTensor> {
    swapaxes(input, axis0, axis1)
}

/// Reorder the dimensions; `dims` must be a permutation of all of them.
#[pyfunction]
#[pyo3(signature = (input, *dims))]
pub fn permute(input: &Bound<PyAny>, dims: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.permute(dims)
}

/// Move dimensions to new positions, keeping the relative order of the rest.
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

/// Alias of `movedim`, under its array-library spelling.
#[pyfunction]
#[pyo3(signature = (input, source, destination))]
pub fn moveaxis(
    input: &Bound<PyAny>,
    source: &Bound<PyAny>,
    destination: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    movedim(input, source, destination)
}

/// Repeat size-1 dimensions out to the given shape.
#[pyfunction]
#[pyo3(signature = (input, *shape))]
pub fn expand(input: &Bound<PyAny>, shape: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.expand(shape)
}

/// Tile the tensor the given number of times along each dimension.
#[pyfunction]
#[pyo3(signature = (input, *repeats))]
pub fn repeat(input: &Bound<PyAny>, repeats: &Bound<PyTuple>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.repeat(repeats)
}

/// Repeat each element in place `repeats` times along `dim`, rather than tiling the whole tensor.
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

/// Reverse the order of elements along the given dimensions.
#[pyfunction]
#[pyo3(signature = (input, dims))]
pub fn flip(input: &Bound<PyAny>, dims: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.flip(dims)
}

/// Shift elements along `dims`, wrapping the ones that fall off the end back to the start.
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

/// Alias of `clamp`, under its array-library spelling.
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

/// Limit every element to `[min, max]`. Either bound may be omitted.
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

// Free-function forms of the tensor methods. Each is the same three lines --
// borrow the operand, call the method of the same name -- so they are generated
// rather than written out 34 times, which is also what kept `abs`, `sqrt`,
// `exp` and `log` missing while `log1p`, `log2`, `log10`, `expm1` and `rsqrt`
// were all present.
// Each forwarder carries the same one-line description as the method it calls.
// The two are written out separately because one is a `#[pymethods]` block and
// the other is generated here, and `test_forwarders_and_methods_describe
// _themselves_identically` fails if they ever drift apart.
macro_rules! unary_forwarders {
    ($($name:ident => $doc:literal),* $(,)?) => {
        $(
            #[doc = $doc]
            #[pyfunction]
            pub fn $name(input: &Bound<PyAny>) -> PyResult<PyTensor> {
                let tensor = borrow_tensor(input)?;
                tensor.$name()
            }
        )*
    };
}

macro_rules! binary_forwarders {
    ($($name:ident => $doc:literal),* $(,)?) => {
        $(
            #[doc = $doc]
            #[pyfunction]
            pub fn $name(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<PyTensor> {
                let tensor = borrow_tensor(input)?;
                tensor.$name(other)
            }
        )*
    };
}
unary_forwarders!(
    abs => "Element-wise absolute value.",
    acos => "Element-wise inverse cosine, returning radians in `[0, pi]`. Inputs outside `[-1, 1]` give NaN.",
    acosh => "Element-wise inverse hyperbolic cosine. Inputs below 1 give NaN.",
    asin => "Element-wise inverse sine, returning radians in `[-pi/2, pi/2]`. Inputs outside `[-1, 1]` give NaN.",
    asinh => "Element-wise inverse hyperbolic sine.",
    atan => "Element-wise inverse tangent, returning radians in `(-pi/2, pi/2)`.",
    atanh => "Element-wise inverse hyperbolic tangent. Inputs outside `(-1, 1)` give NaN or infinity.",
    bitwise_not => "Element-wise bitwise complement, and logical NOT for booleans.",
    ceil => "Round towards positive infinity.",
    cos => "Element-wise cosine, taking radians.",
    cosh => "Element-wise hyperbolic cosine.",
    erf => "Element-wise error function.",
    erfc => "Element-wise complementary error function, `1 - erf(x)`, accurate in the tails where that subtraction would cancel.",
    exp => "Element-wise `e ** x`.",
    expm1 => "Element-wise `exp(x) - 1`, accurate for small `x` where the subtraction would cancel.",
    floor => "Round towards negative infinity.",
    log => "Element-wise natural logarithm. Zero gives `-inf`, negatives give NaN.",
    log10 => "Element-wise base-10 logarithm.",
    log1p => "Element-wise `log(1 + x)`, accurate for small `x` where `log(1 + x)` would cancel.",
    log2 => "Element-wise base-2 logarithm.",
    ravel => "A contiguous 1-D view of every element, in row-major order.",
    reciprocal => "Element-wise `1 / x`.",
    relu => "Element-wise `max(x, 0)`.",
    rsqrt => "Element-wise `1 / sqrt(x)`, computed without the intermediate root.",
    selu => "Scaled Exponential Linear Unit, with the fixed constants that make it self-normalizing.",
    sigmoid => "Element-wise `1 / (1 + exp(-x))`, evaluated so that large-magnitude inputs saturate instead of producing NaN.",
    sign => "-1, 0 or 1 according to each element's sign. NaN gives NaN.",
    silu => "Sigmoid Linear Unit (Swish), `x * sigmoid(x)`.",
    sin => "Element-wise sine, taking radians.",
    sinh => "Element-wise hyperbolic sine.",
    softsign => "Element-wise `x / (1 + abs(x))`.",
    sqrt => "Element-wise square root. Negative inputs give NaN.",
    tan => "Element-wise tangent, taking radians.",
    tanh => "Element-wise hyperbolic tangent.",
);

binary_forwarders!(
    bmm => "Batched matrix product of two 3-D tensors with matching batch sizes.",
    dot => "Inner product of two 1-D tensors.",
    eq => "Element-wise equality, giving a boolean tensor.",
    floor_divide => "Element-wise division rounded towards negative infinity, matching Python's `//`.",
    ge => "Element-wise `>=`, giving a boolean tensor.",
    gt => "Element-wise `>`, giving a boolean tensor.",
    le => "Element-wise `<=`, giving a boolean tensor.",
    logaddexp => "`log(exp(a) + exp(b))`, shifted so neither term overflows.",
    lt => "Element-wise `<`, giving a boolean tensor.",
    matmul => "Matrix product, broadcasting over leading batch dimensions.",
    maximum => "Element-wise larger of two tensors.",
    minimum => "Element-wise smaller of two tensors.",
    ne => "Element-wise inequality, giving a boolean tensor.",
    pow => "Raise each element of the first tensor to the matching power from the second.",
    remainder => "Element-wise modulo taking the sign of the divisor, matching Python's `%`.",
);

/// Raise every element below `min` up to it.
#[pyfunction]
pub fn clamp_min(input: &Bound<PyAny>, min: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clamp_min(min)
}

/// Lower every element above `max` down to it.
#[pyfunction]
pub fn clamp_max(input: &Bound<PyAny>, max: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.clamp_max(max)
}

/// Round to the nearest integer, halves to even.
#[pyfunction]
#[pyo3(signature = (input, decimals=0))]
pub fn round(input: &Bound<PyAny>, decimals: i32) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.round(decimals)
}

/// Split into `sections` equal parts along `dim`.
#[pyfunction]
#[pyo3(signature = (input, chunks, dim=0))]
pub fn chunk(input: &Bound<PyAny>, chunks: usize, dim: isize) -> PyResult<Vec<PyTensor>> {
    let tensor = borrow_tensor(input)?;
    tensor.chunk(chunks, dim)
}

/// Split along `dim` into pieces of the given size, or into the given explicit sizes.
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

/// Take the entries `index` names along `dim`, in the order given.
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

/// Take one element per position, choosing its `dim` coordinate from `index`.
#[pyfunction]
#[pyo3(signature = (input, dim, index))]
pub fn gather(input: &Bound<PyAny>, dim: isize, index: &Bound<PyAny>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let index_tensor = borrow_tensor(index)?;
    tensor.gather(dim, &index_tensor)
}

/// Write `src` into a copy of `input` at the positions `index` names along `dim`.
#[pyfunction]
#[pyo3(signature = (input, dim, index, src))]
pub fn scatter(
    input: &Bound<PyAny>,
    dim: isize,
    index: &Bound<PyAny>,
    src: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let index_tensor = borrow_tensor(index)?;
    let src_tensor = borrow_tensor(src)?;
    tensor.scatter(dim, &index_tensor, &src_tensor)
}

/// Like `scatter`, but adds into the target instead of overwriting, so repeated indices accumulate.
#[pyfunction]
#[pyo3(signature = (input, dim, index, src))]
pub fn scatter_add(
    input: &Bound<PyAny>,
    dim: isize,
    index: &Bound<PyAny>,
    src: &Bound<PyAny>,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let index_tensor = borrow_tensor(index)?;
    let src_tensor = borrow_tensor(src)?;
    tensor.scatter_add(dim, &index_tensor, &src_tensor)
}

/// Pick from `input` where `condition` is true and from `other` where it is false.
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

/// Count occurrences of each non-negative integer value.
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

/// Expand class indices into a trailing axis of `num_classes` indicators.
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

/// Replace the positions `mask` selects with `value`.
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

/// Normalize along `dim` so the values are positive and sum to 1. Shifted by the row maximum, so large inputs do not overflow.
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn softmax(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.softmax(dim)
}

/// Logarithm of `softmax`, computed directly rather than as `log(softmax(x))`, which underflows for confident rows.
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn log_softmax(input: &Bound<PyAny>, dim: Option<isize>) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.log_softmax(dim)
}

/// `softmax` over the positions `mask` leaves alone: a true entry is excluded from the max and the sum -- not zeroed after normalizing -- and comes out 0.
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

/// `log_softmax` over the positions `mask` selects. See `masked_softmax`.
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

/// Sum over `dim`, or over every element when `dim` is omitted.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn sum(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.sum(dim, Some(keepdim))
}

/// Product over `dim`, or over every element when `dim` is omitted.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn prod(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.prod(dim, Some(keepdim))
}

/// Arithmetic mean over `dim`, or over every element when `dim` is omitted.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn mean(input: &Bound<PyAny>, dim: Option<&Bound<PyAny>>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.mean(dim, Some(keepdim))
}

/// Whether every element is true (or non-zero) over `dim`.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn all(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.all(dim, Some(keepdim))
}

/// Whether any element is true (or non-zero) over `dim`.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn any(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.any(dim, Some(keepdim))
}

/// Largest element over `dim`, values only.
///
/// `max(dim=...)` reports the index alongside, and finding it is most of the
/// cost; see `Tensor.amax`. NumPy and PyTorch both spell this `amax`.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn amax(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    borrow_tensor(input)?.max_values(dim, keepdim)
}

/// Smallest element over `dim`, values only. See [`amax`].
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn amin(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    borrow_tensor(input)?.min_values(dim, keepdim)
}

/// Like `amax`, ignoring NaN.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanamax(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    borrow_tensor(input)?.nanmax_values(dim, keepdim)
}

/// Like `amin`, ignoring NaN.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanamin(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    borrow_tensor(input)?.nanmin_values(dim, keepdim)
}

/// Largest element over `dim`; with a `dim` it returns the values and their indices.
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

/// Smallest element over `dim`; with a `dim` it returns the values and their indices.
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

/// Index of the largest element over `dim`. Ties go to the first occurrence.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmax(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.argmax(dim, Some(keepdim))
}

/// Index of the smallest element over `dim`. Ties go to the first occurrence.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn argmin(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.argmin(dim, Some(keepdim))
}

/// Running sum along `dim`, keeping the input's shape.
#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn cumsum(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cumsum(dim)
}

/// Running product along `dim`, keeping the input's shape.
#[pyfunction]
#[pyo3(signature = (input, dim))]
pub fn cumprod(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.cumprod(dim)
}

/// Standard deviation over `dim`. `unbiased` applies Bessel's correction.
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

/// Variance over `dim`. `unbiased` applies Bessel's correction.
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

/// Vector `p`-norm over `dim`, or over every element when `dim` is omitted.
#[pyfunction]
#[pyo3(signature = (input, p=None, dim=None, keepdim=false))]
pub fn norm(
    input: &Bound<PyAny>,
    p: Option<&Bound<PyAny>>,
    dim: Option<&Bound<PyAny>>,
    keepdim: bool,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.norm(p, dim, Some(keepdim))
}

/// `log(sum(exp(x)))` over `dim`, shifted by the maximum so large values do not overflow.
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

/// Like `sum`, treating NaN as zero. An all-NaN slice sums to 0.
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

/// Like `mean`, ignoring NaN. A slice that is entirely NaN gives NaN.
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

/// Like `max`, ignoring NaN.
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

/// Like `min`, ignoring NaN.
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

/// Element-wise test for NaN.
#[pyfunction]
pub fn isnan(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isnan)
}

/// Element-wise test for positive or negative infinity.
#[pyfunction]
pub fn isinf(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isinf)
}

/// Element-wise test for a value that is neither NaN nor infinite.
#[pyfunction]
pub fn isfinite(input: &Bound<PyAny>) -> PyResult<PyTensor> {
    finite_predicate(input, Tensor::isfinite)
}

/// Replace NaN with `nan` and the infinities with `posinf`/`neginf`, defaulting to the dtype's finite extremes.
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

/// Zero out values with magnitude below `lambd`, leaving the rest unchanged.
#[pyfunction]
#[pyo3(signature = (input, lambd=0.5))]
pub fn hardshrink(input: &Bound<PyAny>, lambd: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.hardshrink(Some(lambd))
}

/// Element-wise `log(1 + exp(beta * x)) / beta`, falling back to the linear `x` above `threshold`.
#[pyfunction]
#[pyo3(signature = (input, beta=1.0, threshold=20.0))]
pub fn softplus(input: &Bound<PyAny>, beta: f64, threshold: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.softplus(Some(beta), Some(threshold))
}

/// Gaussian Error Linear Unit, `x * Phi(x)`. Pass `approximate=\"tanh\"` for the tanh approximation.
#[pyfunction]
#[pyo3(signature = (input, approximate="none"))]
pub fn gelu(input: &Bound<PyAny>, approximate: &str) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.gelu(Some(approximate))
}

/// Exponential Linear Unit: `x` where positive, `alpha * (exp(x) - 1)` elsewhere.
#[pyfunction]
#[pyo3(signature = (input, alpha=1.0))]
pub fn elu(input: &Bound<PyAny>, alpha: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.elu(Some(alpha))
}

/// `x` where positive, `negative_slope * x` elsewhere.
///
/// The one activation that had no functional form: `relu`, `elu`, `selu`,
/// `silu`, `gelu`, `softplus`, `hardshrink` and `softsign` all did, and
/// `nn.LeakyReLU` existed as a layer, but `F.leaky_relu` did not.
#[pyfunction]
#[pyo3(signature = (input, negative_slope=0.01))]
pub fn leaky_relu(input: &Bound<PyAny>, negative_slope: f64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.leaky_relu(Some(negative_slope))
}

/// Zero everything below the `diagonal`-th diagonal.
#[pyfunction]
#[pyo3(signature = (input, diagonal=0))]
pub fn triu(input: &Bound<PyAny>, diagonal: i64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.triu(diagonal)
}

/// Zero everything above the `diagonal`-th diagonal.
#[pyfunction]
#[pyo3(signature = (input, diagonal=0))]
pub fn tril(input: &Bound<PyAny>, diagonal: i64) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.tril(diagonal)
}

/// The requested diagonal of the last two dimensions, as a new trailing axis.
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

/// Sum of the main diagonal.
#[pyfunction]
#[pyo3(signature = (input, offset=0, dim1=-2, dim2=-1))]
pub fn trace(input: &Bound<PyAny>, offset: isize, dim1: isize, dim2: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.trace(offset, dim1, dim2)
}

/// Solve the linear system `A x = b` for `x`.
#[pyfunction]
pub fn solve(lhs: &Bound<PyAny>, rhs: &Bound<PyAny>) -> PyResult<PyTensor> {
    let lhs_tensor = borrow_tensor(lhs)?;
    lhs_tensor.solve(rhs)
}

/// The `k` largest elements along `dim`, with their indices. Pass `largest=False` for the smallest.
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

/// Sort along `dim`, returning the sorted values and the indices that produced them.
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

/// The indices that would sort along `dim`.
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

/// Middle element over `dim`. For an even count this is the lower of the two, not their average -- use `quantile(0.5)` for the interpolated definition.
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

/// Like `median`, ignoring NaN.
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=false))]
pub fn nanmedian(input: &Bound<PyAny>, dim: Option<isize>, keepdim: bool) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    tensor.nanmedian(dim, Some(keepdim))
}

/// The `q`-th quantile over `dim`, interpolating between neighbouring elements.
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

/// Like `quantile`, ignoring NaN.
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

/// Normalize over the trailing `normalized_shape` dimensions using that slice's own mean and variance, then scale and shift.
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

/// Root-mean-square layer normalization (RMSNorm) — the normalization used by
/// LLaMA/Mistral/Gemma and most modern LLMs. Normalizes by the RMS over
/// `normalized_shape` (no mean subtraction, no bias) and rescales by `weight`.
#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, eps=1e-6))]
pub fn rms_norm(
    input: &Bound<PyAny>,
    normalized_shape: &Bound<PyAny>,
    weight: Option<&Bound<PyAny>>,
    eps: f64,
) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let shape = parse_normalized_shape(normalized_shape)?;
    let weight_tensor = borrow_optional_tensor(weight)?;
    tensor.rms_norm(shape, weight_tensor.as_deref(), Some(eps))
}

/// Scaled dot-product attention — the core Transformer primitive
/// (Vaswani et al., 2017). Computes `softmax(Q Kᵀ / sqrt(E) + bias) V` over the
/// key axis, fully differentiable through `query`, `key`, `value` and a float
/// `attn_mask`. Leading batch axes broadcast, so multi-head layouts
/// `(batch, heads, seq, dim)` work directly.
///
/// `attn_mask` is broadcastable to the scores `(..., L, S)`: a float mask is
/// added to the scores (use `-inf` to disallow), a bool mask keeps `True`
/// positions and disables `False` ones. `is_causal=True` applies an
/// autoregressive mask (position i attends only to j <= i); combining it with an
/// explicit `attn_mask` is rejected. `scale` overrides the default `1/sqrt(E)`.
#[pyfunction]
#[pyo3(signature = (query, key, value, attn_mask=None, is_causal=false, scale=None))]
pub fn scaled_dot_product_attention(
    query: &Bound<PyAny>,
    key: &Bound<PyAny>,
    value: &Bound<PyAny>,
    attn_mask: Option<&Bound<PyAny>>,
    is_causal: bool,
    scale: Option<f64>,
) -> PyResult<PyTensor> {
    let q = borrow_tensor(query)?;
    let k = borrow_tensor(key)?;
    let v = borrow_tensor(value)?;
    let mask = borrow_optional_tensor(attn_mask)?;
    let result = engine::ops::scaled_dot_product_attention(
        q.tensor(),
        k.tensor(),
        v.tensor(),
        mask.as_deref().map(|m| m.tensor()),
        is_causal,
        scale,
    )
    .map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Rotary Position Embedding (RoPE; Su et al., 2021) — the positional encoding
/// of LLaMA/Mistral/Qwen and most modern LLMs. Rotates pairs of features of `x`
/// (shape `(..., seq, head_dim)`, even `head_dim`) by position-dependent angles,
/// injecting relative position with no learned parameters. `offset` shifts the
/// starting position (KV-cache decoding); `base` sets the frequency spectrum.
#[pyfunction]
#[pyo3(signature = (x, base=10000.0, offset=0))]
pub fn rope(x: &Bound<PyAny>, base: f64, offset: usize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(x)?;
    let result = engine::ops::rope(tensor.tensor(), base, offset).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Gated Linear Unit (Dauphin et al., 2017). Splits `input` in half along `dim`
/// into `(a, b)` and returns `a * sigmoid(b)` — the gate underlying GLU-family
/// feed-forward blocks (GEGLU, SwiGLU). `dim` must have even length.
#[pyfunction]
#[pyo3(signature = (input, dim=-1))]
pub fn glu(input: &Bound<PyAny>, dim: isize) -> PyResult<PyTensor> {
    let tensor = borrow_tensor(input)?;
    let result = engine::ops::glu(tensor.tensor(), dim).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// Join tensors along an existing dimension. Every other dimension must match.
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
    PyTensor::concatenate(tensors, Some(dim))
}

/// Join tensors along a new dimension, which all of them must be shaped alike for.
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn stack(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
    PyTensor::stack(tensors, Some(dim))
}

/// Element-wise test for being within `rtol`/`atol`.
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

/// Whether two tensors have the same shape and every element equal. NaN is never equal to itself.
#[pyfunction]
pub fn array_equal(input: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<bool> {
    let lhs = PyTensor::from_python_value(input)?;
    let rhs = PyTensor::from_python_value(other)?;
    lhs.array_equal(&rhs)
}

/// Whether every pair of elements is within `rtol`/`atol`.
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
    parent.add_function(wrap_pyfunction!(scatter, parent)?)?;
    parent.add_function(wrap_pyfunction!(scatter_add, parent)?)?;
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
    parent.add_function(wrap_pyfunction!(amax, parent)?)?;
    parent.add_function(wrap_pyfunction!(amin, parent)?)?;
    parent.add_function(wrap_pyfunction!(argmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(argmin, parent)?)?;
    parent.add_function(wrap_pyfunction!(cumsum, parent)?)?;
    parent.add_function(wrap_pyfunction!(cumprod, parent)?)?;
    parent.add_function(wrap_pyfunction!(std_fn, parent)?)?;
    parent.add_function(wrap_pyfunction!(var, parent)?)?;
    parent.add_function(wrap_pyfunction!(norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(logsumexp, parent)?)?;
    parent.add_function(wrap_pyfunction!(nansum, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmean, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmax, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanmin, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanamax, parent)?)?;
    parent.add_function(wrap_pyfunction!(nanamin, parent)?)?;
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
    parent.add_function(wrap_pyfunction!(leaky_relu, parent)?)?;
    parent.add_function(wrap_pyfunction!(selu, parent)?)?;
    parent.add_function(wrap_pyfunction!(silu, parent)?)?;
    parent.add_function(wrap_pyfunction!(softsign, parent)?)?;
    parent.add_function(wrap_pyfunction!(tanh, parent)?)?;
    parent.add_function(wrap_pyfunction!(log2, parent)?)?;
    parent.add_function(wrap_pyfunction!(log10, parent)?)?;
    parent.add_function(wrap_pyfunction!(erf, parent)?)?;
    parent.add_function(wrap_pyfunction!(erfc, parent)?)?;
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
    parent.add_function(wrap_pyfunction!(abs, parent)?)?;
    parent.add_function(wrap_pyfunction!(sqrt, parent)?)?;
    parent.add_function(wrap_pyfunction!(exp, parent)?)?;
    parent.add_function(wrap_pyfunction!(log, parent)?)?;
    parent.add_function(wrap_pyfunction!(pow, parent)?)?;
    parent.add_function(wrap_pyfunction!(bitwise_not, parent)?)?;
    parent.add_function(wrap_pyfunction!(matmul, parent)?)?;
    parent.add_function(wrap_pyfunction!(eq, parent)?)?;
    parent.add_function(wrap_pyfunction!(ne, parent)?)?;
    parent.add_function(wrap_pyfunction!(lt, parent)?)?;
    parent.add_function(wrap_pyfunction!(le, parent)?)?;
    parent.add_function(wrap_pyfunction!(gt, parent)?)?;
    parent.add_function(wrap_pyfunction!(ge, parent)?)?;
    parent.add_function(wrap_pyfunction!(floor_divide, parent)?)?;
    parent.add_function(wrap_pyfunction!(remainder, parent)?)?;
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
    parent.add_function(wrap_pyfunction!(rms_norm, parent)?)?;
    parent.add_function(wrap_pyfunction!(scaled_dot_product_attention, parent)?)?;
    parent.add_function(wrap_pyfunction!(rope, parent)?)?;
    parent.add_function(wrap_pyfunction!(glu, parent)?)?;
    parent.add_function(wrap_pyfunction!(cat, parent)?)?;
    parent.add_function(wrap_pyfunction!(stack, parent)?)?;
    parent.add_function(wrap_pyfunction!(dot, parent)?)?;
    parent.add_function(wrap_pyfunction!(bmm, parent)?)?;
    parent.add_function(wrap_pyfunction!(isclose, parent)?)?;
    parent.add_function(wrap_pyfunction!(array_equal, parent)?)?;
    parent.add_function(wrap_pyfunction!(allclose, parent)?)?;
    Ok(())
}
