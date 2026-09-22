// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

// Child modules hosting the PyTensor method impls and interop helpers. They
// are children (not siblings) so they retain access to the private `inner`
// field and inherit this file's imports via `use super::*`.
#[path = "pytensor/arithmetic.rs"]
mod arithmetic;
#[path = "pytensor/bitwise.rs"]
mod bitwise;
#[path = "pytensor/comparison.rs"]
mod comparison;
#[path = "pytensor/comparison_dunder.rs"]
mod comparison_dunder;
#[path = "pytensor/concat_split.rs"]
mod concat_split;
#[path = "pytensor/creation/basic.rs"]
mod creation_basic;
#[path = "pytensor/creation/like.rs"]
mod creation_like;
#[path = "pytensor/creation/range.rs"]
mod creation_range;
#[path = "pytensor/grad.rs"]
mod grad;
#[path = "pytensor/isclose.rs"]
mod isclose;
#[path = "pytensor/math.rs"]
mod math;
#[path = "pytensor/operations.rs"]
mod operations;
#[path = "pytensor/properties.rs"]
mod properties;
#[path = "python/args.rs"]
mod py_args;
#[path = "python/convert.rs"]
mod py_convert;
#[path = "pytensor/numpy.rs"]
mod py_numpy;
#[path = "python/numpy.rs"]
mod py_numpy_interop;
#[path = "pytensor/reduction.rs"]
mod reduction;
#[path = "pytensor/repr.rs"]
mod repr_impl;

pub(crate) use self::py_args::*;
pub(crate) use self::py_convert::*;
pub(crate) use self::py_numpy_interop::*;

use crate::device::{PyDevice, resolve_device, resolve_device_or};
use crate::dtype;
use crate::error::_convert_error;
use crate::numpy_compat::cross_impl;
use engine::nn;
use engine::ops::binary::{BinaryOpKind, coerce_binary_operands};
use engine::ops::reduction::QuantileInterpolation;
use engine::ops::shape_ops::RepeatInterleaveSpec;
use engine::random;
use engine::tensor::{Shape, TensorData};
use engine::{DataType, Device, MinitensorError, Tensor, TensorIndex};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyDict, PyInt, PyList, PyModule, PySequence, PySequenceMethods, PySlice,
    PyString, PyTuple,
};
use pyo3::{Py, PyRefMut};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::OnceLock;

fn register_leaf_tensor(tensor: &Tensor) {
    if tensor.requires_grad() && tensor.grad_fn().is_none() {
        let _ = engine::autograd::add_to_graph(tensor, None);
    }
}

/// The `PyTensor` behind a value that is one, or that wraps one as `_tensor`.
///
/// A `PyRef` rather than an owned `PyTensor`, because extracting the class by
/// value clones it, and a `Tensor` clone allocates twice -- once for the shape
/// and once for the strides. Most callers only read, and the ones that need to
/// own clone once from here instead of cloning a clone.
///
/// `hasattr` and not `getattr` for the wrapper case: the miss is the common
/// case -- every scalar operand of every binary op reaches here -- and a
/// failed `getattr` raises, which means building and discarding a Python
/// exception on a path that is only asking a question.
pub(crate) fn borrow_wrapped_tensor<'py>(
    value: &'py Bound<'py, PyAny>,
) -> Option<PyRef<'py, PyTensor>> {
    if let Ok(py_tensor) = value.extract::<PyRef<PyTensor>>() {
        return Some(py_tensor);
    }

    let attr_name = intern!(value.py(), "_tensor");
    if value.hasattr(attr_name).ok()?
        && let Ok(inner_attr) = value.getattr(attr_name)
        && let Ok(py_tensor) = inner_attr.extract::<PyRef<PyTensor>>()
    {
        return Some(py_tensor);
    }

    None
}

/// Run `body` on integer indices taken from an integer tensor or any Python
/// sequence of ints (list, tuple, numpy array, ...).
///
/// A callback rather than a returned `Vec` so that the common case costs
/// nothing: an `int64` index tensor already holds exactly what the kernel
/// wants, and its buffer is handed over where it lies. Copying it first is not
/// free at the sizes this is called with -- `x[perm]` over a million positions
/// spent more time narrowing the index than gathering the data.
fn with_index_vector<R>(
    indices: &Bound<PyAny>,
    body: impl FnOnce(&[i64]) -> PyResult<R>,
) -> PyResult<R> {
    let Some(py_tensor) = borrow_wrapped_tensor(indices) else {
        return body(&indices.extract::<Vec<i64>>()?);
    };
    let tensor = py_tensor.inner.contiguous().map_err(_convert_error)?;
    if tensor.ndim() > 1 {
        return Err(PyValueError::new_err(
            "index tensor must be 0-D or 1-D".to_string(),
        ));
    }
    match tensor.dtype() {
        DataType::Int32 => body(
            &tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| PyRuntimeError::new_err("failed to read index tensor data"))?
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<i64>>(),
        ),
        DataType::Int64 => body(
            tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| PyRuntimeError::new_err("failed to read index tensor data"))?,
        ),
        dtype => Err(PyTypeError::new_err(format!(
            "index tensor must have an integer dtype, got {dtype:?}",
        ))),
    }
}

fn parse_clip_bound(value: Option<&Bound<PyAny>>, name: &str) -> PyResult<Option<f64>> {
    match value {
        None => Ok(None),
        Some(bound) => {
            if bound.is_none() {
                return Ok(None);
            }

            if let Ok(val) = bound.extract::<f64>() {
                Ok(Some(val))
            } else if let Ok(int_val) = bound.extract::<i64>() {
                Ok(Some(int_val as f64))
            } else {
                Err(PyTypeError::new_err(format!(
                    "{name} must be a real number or None",
                )))
            }
        }
    }
}

fn extract_real_scalar(value: &Bound<PyAny>, name: &str) -> PyResult<f64> {
    if let Ok(boolean) = value.extract::<bool>() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }

    if let Ok(int_val) = value.extract::<i64>() {
        return Ok(int_val as f64);
    }

    if let Ok(float_val) = value.extract::<f64>() {
        return Ok(float_val);
    }

    Err(PyTypeError::new_err(format!(
        "{name} must be a real number or boolean",
    )))
}

fn parse_quantile_interpolation(mode: Option<&str>) -> PyResult<QuantileInterpolation> {
    let mode = mode.unwrap_or("linear");
    if mode.eq_ignore_ascii_case("linear") {
        Ok(QuantileInterpolation::Linear)
    } else if mode.eq_ignore_ascii_case("lower") {
        Ok(QuantileInterpolation::Lower)
    } else if mode.eq_ignore_ascii_case("higher") {
        Ok(QuantileInterpolation::Higher)
    } else if mode.eq_ignore_ascii_case("midpoint") {
        Ok(QuantileInterpolation::Midpoint)
    } else if mode.eq_ignore_ascii_case("nearest") {
        Ok(QuantileInterpolation::Nearest)
    } else {
        Err(PyValueError::new_err(format!(
            "Invalid interpolation mode '{mode}'. Expected one of: linear, lower, higher, midpoint, nearest",
        )))
    }
}

enum QuantileArg {
    Scalar(f64),
    Multiple(Vec<f64>),
}

fn parse_quantile_arg(q: &Bound<PyAny>) -> PyResult<QuantileArg> {
    if let Ok(value) = q.extract::<f64>() {
        return Ok(QuantileArg::Scalar(value));
    }

    if let Ok(values) = q.extract::<Vec<f64>>() {
        if values.is_empty() {
            Err(PyValueError::new_err(
                "quantile() expected at least one probability value",
            ))
        } else {
            Ok(QuantileArg::Multiple(values))
        }
    } else {
        Err(PyTypeError::new_err(
            "q must be a float or a sequence of floats",
        ))
    }
}

#[pyclass(name = "Shape", module = "minitensor._core", from_py_object)]
#[derive(Clone, Debug)]
pub struct ShapeSequence {
    dims: Vec<usize>,
}

impl ShapeSequence {
    pub fn from_dims<D: Into<Vec<usize>>>(dims: D) -> Self {
        Self { dims: dims.into() }
    }
}

#[pymethods]
impl ShapeSequence {
    #[new]
    fn py_new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Shape({:?})", self.dims))
    }

    fn __len__(&self) -> usize {
        self.dims.len()
    }

    fn __getitem__(&self, index: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = index.py();
        if let Ok(idx) = index.extract::<isize>() {
            let len = self.dims.len() as isize;
            let resolved = if idx < 0 { idx + len } else { idx };
            if resolved < 0 || resolved >= len {
                Err(PyIndexError::new_err("Shape index out of range"))
            } else {
                let value = self.dims[resolved as usize];
                let py_value = i64::try_from(value)
                    .map_err(|_| PyValueError::new_err("Shape dimension too large"))?;
                Ok(PyInt::new(py, py_value).into())
            }
        } else if let Ok(slice) = index.cast::<PySlice>() {
            let indices = slice.indices(self.dims.len() as isize)?;
            let mut values = Vec::with_capacity(indices.slicelength);
            let mut current = indices.start;
            for _ in 0..indices.slicelength {
                values.push(self.dims[current as usize]);
                current += indices.step;
            }
            Ok(Py::new(py, ShapeSequence::from_dims(values))?.into())
        } else {
            Err(PyTypeError::new_err(
                "Shape indices must be integers or slices",
            ))
        }
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(other_shape) = other.extract::<ShapeSequence>() {
            return Ok(self.dims == other_shape.dims);
        }

        if let Ok(other_vec) = other.extract::<Vec<usize>>() {
            return Ok(self.dims == other_vec);
        }

        Ok(false)
    }

    fn to_list(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn to_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.dims)
    }
}

/// An n-dimensional array with optional gradient tracking.
///
/// Construct from a NumPy array or nested sequence, or with one of the
/// creation functions (`zeros`, `randn`, `arange`, ...). Operations record
/// onto the autograd graph when any operand has `requires_grad`.
#[pyclass(name = "Tensor", module = "minitensor._core", from_py_object)]
#[derive(Clone)]
pub struct PyTensor {
    inner: Tensor,
}

impl PyTensor {
    /// Get reference to inner tensor
    pub fn tensor(&self) -> &Tensor {
        &self.inner
    }

    /// Get mutable reference to inner tensor
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.inner
    }

    /// Create from inner tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        // The engine's kernels read tensor storage in contiguous logical
        // order, so a non-contiguous tensor must be materialised before it
        // becomes visible to Python; otherwise every downstream operation
        // would read the wrong elements. The engine no longer produces one
        // (`expand` materialises its broadcast axes), so this is a cheap
        // guard against the invariant regressing, not a hot path.
        let tensor = if tensor.is_contiguous() {
            tensor
        } else {
            match tensor.contiguous() {
                Ok(contiguous) => contiguous,
                Err(_) => tensor,
            }
        };
        register_leaf_tensor(&tensor);
        Self { inner: tensor }
    }

    pub fn from_python_value(value: &Bound<PyAny>) -> PyResult<Self> {
        Self::from_python_value_with_dtype(value, dtype::default_dtype())
    }

    pub fn from_python_value_with_dtype(value: &Bound<PyAny>, dtype: DataType) -> PyResult<Self> {
        if let Some(py_tensor) = borrow_wrapped_tensor(value) {
            return Ok(Self::from_tensor(py_tensor.inner.clone()));
        }

        let tensor = convert_python_data_to_tensor(value, dtype, Device::cpu(), false)?;
        Ok(Self::from_tensor(tensor))
    }

    pub fn infer_python_dtype(value: &Bound<PyAny>) -> Option<DataType> {
        infer_python_value_dtype(value)
    }

    pub fn max_values(&self, dim: Option<Vec<isize>>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.amax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn nanmax_values(&self, dim: Option<Vec<isize>>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.nanamax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn min_values(&self, dim: Option<Vec<isize>>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.amin(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn nanmin_values(&self, dim: Option<Vec<isize>>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.nanamin(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn median_with_indices(
        &self,
        dim: Option<isize>,
        keepdim: bool,
    ) -> PyResult<(Self, Option<Self>)> {
        match self.inner.median(dim, keepdim) {
            Ok((values, indices_opt)) => {
                let values_tensor = Self::from_tensor(values);
                let indices_tensor = indices_opt.map(Self::from_tensor);
                Ok((values_tensor, indices_tensor))
            }
            // Deliberate departure from `_convert_error`, which maps
            // `InvalidArgument` to `ValueError`: `median()` on an empty tensor
            // raises `RuntimeError`, and
            // `test_median_empty_tensor_raises` pins that. Everything else
            // keeps the standard mapping.
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }
}
