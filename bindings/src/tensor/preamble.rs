// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::dtype;
use crate::error::_convert_error;
use crate::numpy_compat::cross_impl;
use engine::nn;
use engine::operations::binary::{BinaryOpKind, coerce_binary_operands};
use engine::operations::reduction::QuantileInterpolation;
use engine::operations::shape_ops::RepeatInterleaveSpec;
use engine::random;
use engine::tensor::{Shape, TensorData};
use engine::{DataType, Device, MinitensorError, Tensor, TensorIndex};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use once_cell::sync::OnceCell;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
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

fn register_leaf_tensor(tensor: &Tensor) {
    if tensor.requires_grad() && tensor.grad_fn().is_none() {
        let _ = engine::autograd::add_to_graph(tensor, None);
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

#[pyclass(name = "Shape", module = "minitensor._core")]
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

/// Python wrapper for Tensor
#[pyclass(name = "Tensor", module = "minitensor._core")]
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
        register_leaf_tensor(&tensor);
        Self { inner: tensor }
    }

    pub fn from_python_value(value: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(py_tensor) = value.extract::<PyTensor>() {
            return Ok(py_tensor);
        }

        if let Ok(inner_attr) = value.getattr(intern!(value.py(), "_tensor"))
            && let Ok(py_tensor) = inner_attr.extract::<PyTensor>()
        {
            return Ok(py_tensor);
        }

        let tensor =
            convert_python_data_to_tensor(value, dtype::default_dtype(), Device::cpu(), false)?;
        Ok(Self::from_tensor(tensor))
    }

    pub fn max_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.max(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn nanmax_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.nanmax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn min_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.min(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn nanmin_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.nanmin(dim, keepdim).map_err(_convert_error)?;
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
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }
}
