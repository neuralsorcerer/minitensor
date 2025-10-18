// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::dtype;
use crate::error::_convert_error;
use crate::numpy_compat::cross_impl;
use engine::operations::binary::{BinaryOpKind, coerce_binary_operands};
use engine::operations::shape_ops::RepeatInterleaveSpec;
use engine::tensor::{Shape, TensorData};
use engine::{DataType, Device, MinitensorError, Tensor, TensorIndex};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use once_cell::sync::OnceCell;
use pyo3::Py;
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
use std::borrow::Cow;
use std::convert::TryFrom;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

fn register_leaf_tensor(tensor: &Tensor) {
    if tensor.requires_grad() && tensor.grad_fn().is_none() {
        let _ = engine::autograd::add_to_graph(tensor, None);
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
        } else if let Ok(slice) = index.downcast::<PySlice>() {
            let indices = slice.indices(self.dims.len() as isize)?;
            let mut values = Vec::with_capacity(indices.slicelength as usize);
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

        if let Ok(inner_attr) = value.getattr(intern!(value.py(), "_tensor")) {
            if let Ok(py_tensor) = inner_attr.extract::<PyTensor>() {
                return Ok(py_tensor);
            }
        }

        let tensor =
            convert_python_data_to_tensor(value, dtype::default_dtype(), Device::cpu(), false)?;
        Ok(Self::from_tensor(tensor))
    }

    pub fn max_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.max(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn min_values(&self, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        let result = self.inner.min(dim, keepdim).map_err(_convert_error)?;
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

#[pymethods]
impl PyTensor {
    #[classattr]
    fn __array_priority__() -> f64 {
        1000.0
    }

    /// Create a new tensor from Python data
    #[new]
    #[pyo3(signature = (data=None, dtype=None, device=None, requires_grad=false))]
    fn new(
        data: Option<&Bound<PyAny>>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        if let Some(value) = data {
            let tensor = convert_python_data_to_tensor(value, dtype, device, requires_grad)?;
            Ok(Self::from_tensor(tensor))
        } else {
            let tensor = Tensor::empty(Shape::new(Vec::new()), dtype, device, requires_grad);
            Ok(Self::from_tensor(tensor))
        }
    }

    // Properties
    #[getter]
    pub fn shape(&self) -> ShapeSequence {
        ShapeSequence::from_dims(self.inner.shape().dims().to_vec())
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        dtype::dtype_to_python_string(self.inner.dtype()).to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn _tensor(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[setter]
    #[allow(non_snake_case)]
    fn set__tensor(&mut self, value: &PyTensor) {
        self.inner = value.inner.clone();
    }

    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[getter]
    fn has_grad(&self) -> bool {
        if engine::autograd::get_gradient(&self.inner).is_some() {
            return true;
        }

        self.inner.has_grad() || self.inner.grad().is_some()
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Self>> {
        if let Some(grad) = engine::autograd::get_gradient(&self.inner) {
            return Ok(Some(Self::from_tensor(grad)));
        }

        if let Some(stored) = self.inner.grad() {
            return Ok(Some(Self::from_tensor((**stored).clone())));
        }

        Ok(None)
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        match self.inner.dtype() {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float64 | DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Get memory usage in bytes
    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.strides().as_slice())
    }

    // Basic tensor info methods
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    // Tensor manipulation methods
    #[pyo3(signature = (*shape))]
    pub fn reshape(&self, shape: &Bound<PyTuple>) -> PyResult<Self> {
        let dims = normalize_variadic_isize_args(shape, "shape")?;
        let reshaped = engine::operations::reshape_with_inference(&self.inner, dims)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(reshaped))
    }

    #[pyo3(signature = (*shape))]
    pub fn view(&self, shape: &Bound<PyTuple>) -> PyResult<Self> {
        self.reshape(shape)
    }

    #[pyo3(signature = (dim0=0, dim1=1))]
    pub fn transpose(&self, dim0: Option<isize>, dim1: Option<isize>) -> PyResult<Self> {
        let dim0 = dim0.unwrap_or(0);
        let dim1 = dim1.unwrap_or(1);
        let result = self.inner.transpose(dim0, dim1).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*dims))]
    pub fn permute(&self, dims: &Bound<PyTuple>) -> PyResult<Self> {
        let dims_vec = normalize_variadic_isize_args(dims, "dims")?;
        let result = self.inner.permute(dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn movedim(&self, source: &Bound<PyAny>, destination: &Bound<PyAny>) -> PyResult<Self> {
        let src_vec: Vec<isize> = match source.extract::<isize>() {
            Ok(v) => vec![v],
            Err(_) => source.extract()?,
        };
        let dst_vec: Vec<isize> = match destination.extract::<isize>() {
            Ok(v) => vec![v],
            Err(_) => destination.extract()?,
        };
        let result = engine::operations::shape_ops::movedim(&self.inner, &src_vec, &dst_vec)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(name = "moveaxis")]
    #[pyo3(signature = (source, destination))]
    pub fn moveaxis_alias(
        &self,
        source: &Bound<PyAny>,
        destination: &Bound<PyAny>,
    ) -> PyResult<Self> {
        self.movedim(source, destination)
    }

    #[pyo3(name = "swapaxes")]
    #[pyo3(signature = (dim0, dim1))]
    pub fn swapaxes_alias(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        self.transpose(Some(dim0), Some(dim1))
    }

    #[pyo3(name = "swapdims")]
    #[pyo3(signature = (dim0, dim1))]
    pub fn swapdims_alias(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        self.transpose(Some(dim0), Some(dim1))
    }

    #[pyo3(signature = (dim=None))]
    pub fn squeeze(&self, dim: Option<isize>) -> PyResult<Self> {
        let result = if let Some(d) = dim {
            self.inner.squeeze_dim(d)
        } else {
            self.inner.squeeze()
        }
        .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.unsqueeze(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*dims))]
    pub fn expand(&self, dims: &Bound<PyTuple>) -> PyResult<Self> {
        let dims_vec = normalize_variadic_isize_args(dims, "shape")?;
        let result = self.inner.expand(dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*repeats))]
    pub fn repeat(&self, repeats: &Bound<PyTuple>) -> PyResult<Self> {
        let repeats_any = if repeats.len() == 1 {
            let first = repeats.get_item(0)?;
            if first.downcast::<PySequence>().is_ok() {
                first.clone().into_any()
            } else {
                repeats.clone().into_any()
            }
        } else {
            repeats.clone().into_any()
        };
        let repeat_vec = normalize_repeat_spec(&repeats_any)?;
        let result = self.inner.repeat(repeat_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn flip(&self, dims: &Bound<PyAny>) -> PyResult<Self> {
        let dims_vec = normalize_required_axes(dims, "dims")?;
        let result =
            engine::operations::shape_ops::flip(&self.inner, &dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (shifts, dims=None))]
    pub fn roll(&self, shifts: &Bound<PyAny>, dims: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let shift_vec = normalize_roll_shifts(shifts)?;
        let dims_vec = normalize_optional_axes(dims)?;
        let dims_ref = dims_vec.as_ref().map(|d| d.as_slice());
        let result = engine::operations::shape_ops::roll(&self.inner, &shift_vec, dims_ref)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (repeats, dim=None, output_size=None))]
    pub fn repeat_interleave(
        &self,
        repeats: &Bound<PyAny>,
        dim: Option<isize>,
        output_size: Option<usize>,
    ) -> PyResult<Self> {
        if let Ok(value) = repeats.extract::<usize>() {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Scalar(value),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(seq) = repeats.extract::<Vec<i64>>() {
            let mut converted = Vec::with_capacity(seq.len());
            for value in seq {
                if value < 0 {
                    return Err(PyValueError::new_err(
                        "repeat_interleave: repeats must be non-negative integers",
                    ));
                }
                let value = usize::try_from(value).map_err(|_| {
                    PyValueError::new_err("repeat_interleave: repeat value exceeds platform limits")
                })?;
                converted.push(value);
            }
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Slice(&converted),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(py_tensor) = repeats.extract::<PyRef<PyTensor>>() {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Tensor(py_tensor.tensor()),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(bound_attr) = repeats.getattr("_tensor") {
            if let Ok(py_tensor) = bound_attr.extract::<PyRef<PyTensor>>() {
                let result = engine::operations::shape_ops::repeat_interleave(
                    &self.inner,
                    RepeatInterleaveSpec::Tensor(py_tensor.tensor()),
                    dim,
                    output_size,
                )
                .map_err(_convert_error)?;
                return Ok(Self::from_tensor(result));
            }
        }

        Err(PyTypeError::new_err(
            "repeat_interleave: repeats must be an int, sequence of ints, or Tensor",
        ))
    }

    #[pyo3(signature = (dim, start, length))]
    pub fn narrow(&self, dim: isize, start: usize, length: usize) -> PyResult<Self> {
        let result = engine::operations::shape_ops::narrow(&self.inner, dim, start, length)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    pub fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let result = self
            .inner
            .flatten(start_dim, end_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn ravel(&self) -> PyResult<Self> {
        self.flatten(0, -1)
    }

    // Tensor operations
    fn clone(&self) -> PyResult<Self> {
        let result = self.inner.deep_clone().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    fn detach_(&mut self) {
        self.inner.detach_inplace();
    }

    fn contiguous(&self) -> PyResult<Self> {
        let result = self.inner.contiguous().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn to(&self, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> PyResult<Self> {
        let mut dtype_spec: Option<DataType> = None;
        let mut device_spec: Option<Device> = None;

        if let Some(mapping) = kwargs {
            for (key, value) in mapping.iter() {
                let key_string = key.str()?.to_str()?.to_owned();
                match key_string.as_str() {
                    "dtype" => {
                        if !value.is_none() {
                            dtype_spec = Some(parse_dtype_like(&value)?);
                        }
                    }
                    "device" => {
                        if !value.is_none() {
                            device_spec = Some(parse_device_like(&value)?);
                        }
                    }
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "to() got an unexpected keyword argument '{key_string}'"
                        )));
                    }
                }
            }
        }

        if args.len() > 1 {
            return Err(PyTypeError::new_err(format!(
                "to() takes at most 1 positional argument but {} were given",
                args.len()
            )));
        }

        if args.len() == 1 {
            let arg0 = args.get_item(0)?;
            if arg0.is_none() {
                // Explicit None does nothing
            } else if let Ok(py_device) = arg0.extract::<PyDevice>() {
                if device_spec.is_some() {
                    return Err(PyTypeError::new_err(
                        "to() received multiple device specifications",
                    ));
                }
                device_spec = Some(py_device.device());
            } else if let Ok(string_value) = arg0.extract::<String>() {
                match dtype::parse_dtype(&string_value) {
                    Ok(dtype) => {
                        if let Some(existing) = dtype_spec {
                            if existing != dtype {
                                return Err(PyTypeError::new_err(
                                    "dtype specified both positionally and via keyword",
                                ));
                            }
                        }
                        dtype_spec = Some(dtype);
                    }
                    Err(_) => {
                        let device = Device::from_str(&string_value).map_err(|err| {
                            PyValueError::new_err(format!(
                                "Unsupported device specification '{string_value}': {err}"
                            ))
                        })?;
                        if device_spec.is_some() {
                            return Err(PyTypeError::new_err(
                                "to() received multiple device specifications",
                            ));
                        }
                        device_spec = Some(device);
                    }
                }
            } else {
                return Err(PyTypeError::new_err(
                    "to() expects dtype strings, device strings, or Device objects",
                ));
            }
        }

        let mut result = self.inner.clone();
        let mut mutated = false;

        if let Some(dtype) = dtype_spec {
            if result.dtype() != dtype {
                result = result.astype(dtype).map_err(_convert_error)?;
                mutated = true;
            }
        }

        if let Some(device) = device_spec {
            if result.device() != device {
                result = result.to(device).map_err(_convert_error)?;
                mutated = true;
            }
        }

        if mutated {
            Ok(Self::from_tensor(result))
        } else {
            Ok(Self {
                inner: self.inner.clone(),
            })
        }
    }

    fn cpu(&self) -> PyResult<Self> {
        let result = self.inner.to(Device::cpu()).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn astype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = dtype::parse_dtype(dtype)?;
        let result = self.inner.astype(dtype).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    // Gradient operations
    #[pyo3(signature = (gradient=None, retain_graph=false, create_graph=false))]
    fn backward(
        &self,
        gradient: Option<&Bound<PyAny>>,
        retain_graph: bool,
        create_graph: bool,
    ) -> PyResult<()> {
        if create_graph {
            return Err(PyNotImplementedError::new_err(
                "create_graph=True is not supported; all computations execute in the Rust backend",
            ));
        }

        if !self.requires_grad() && self.is_leaf() {
            return Err(PyRuntimeError::new_err(
                "element 0 of tensors does not require grad and does not have a grad_fn",
            ));
        }

        if !retain_graph && engine::autograd::is_graph_consumed() {
            return Err(PyRuntimeError::new_err(
                "Computation graph has been freed. Re-run the forward pass or call backward(retain_graph=True).",
            ));
        }

        let grad_tensor = if let Some(value) = gradient {
            if value.is_none() {
                None
            } else if let Ok(py_tensor) = value.extract::<PyTensor>() {
                let mut tensor = py_tensor.inner.clone();
                ensure_backward_gradient_compatible(&self.inner, &mut tensor)?;
                Some(tensor)
            } else {
                let mut tensor = tensor_from_py_value(&self.inner, value)?;
                ensure_backward_gradient_compatible(&self.inner, &mut tensor)?;
                Some(tensor)
            }
        } else {
            None
        };

        self.inner.backward(grad_tensor).map_err(_convert_error)?;

        if !retain_graph {
            engine::autograd::mark_graph_consumed();
        }

        Ok(())
    }

    pub fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        self.inner = self.inner.clone().requires_grad_(requires_grad);
        Ok(())
    }

    #[pyo3(signature = (set_to_none=false))]
    fn zero_grad(&mut self, set_to_none: bool) {
        self.inner.zero_grad(set_to_none);
    }

    // Arithmetic operations
    fn __neg__(&self) -> PyResult<Self> {
        use engine::operations::arithmetic::neg;
        let result = neg(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.add(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, true, BinaryOpKind::Add)?;
        let result = lhs.add(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::sub;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Sub)?;
        let result = sub(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::sub;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, true, BinaryOpKind::Sub)?;
        let result = sub(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::mul;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Mul)?;
        let result = mul(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::mul;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, true, BinaryOpKind::Mul)?;
        let result = mul(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __truediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::div;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rtruediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::operations::arithmetic::div;
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, true, BinaryOpKind::Div)?;
        let result = div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    // Comparison operators as Python dunder methods
    fn __eq__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.eq_from_py(other)
    }

    fn __ne__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ne_from_py(other)
    }

    fn __lt__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.lt_from_py(other)
    }

    fn __le__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.le_from_py(other)
    }

    fn __gt__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.gt_from_py(other)
    }

    fn __ge__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ge_from_py(other)
    }

    pub fn matmul(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.matmul(&other_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn dot(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.dot(&other_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn triu(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.triu(diagonal).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn tril(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.tril(diagonal).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (offset=0, dim1=-2, dim2=-1))]
    pub fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<Self> {
        let result = self
            .inner
            .diagonal(offset, dim1, dim2)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (offset=0, dim1=-2, dim2=-1))]
    pub fn trace(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<Self> {
        let result = self
            .inner
            .trace(offset, dim1, dim2)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(name = "where")]
    pub fn where_method(&self, condition: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<Self> {
        let device = self.inner.device();
        let condition_tensor = tensor_bool_from_py(condition, device)?;

        let other_input = tensor_from_py_value(&self.inner, other)?;
        let (input_cast, other_cast, _) =
            coerce_binary_operands(&self.inner, &other_input, BinaryOpKind::Add)
                .map_err(_convert_error)?;

        let input_tensor = match input_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let other_tensor = match other_cast {
            Cow::Borrowed(_) => other_input,
            Cow::Owned(tensor) => tensor,
        };

        let result = input_tensor
            .where_select(&condition_tensor, &other_tensor)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn masked_fill(&self, mask: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<Self> {
        let device = self.inner.device();
        let mask_tensor = tensor_bool_from_py(mask, device)?;

        let mut tensor_value = tensor_from_py_value(&self.inner, value).map_err(|_| {
            PyTypeError::new_err("masked_fill value must be a Tensor or numeric scalar")
        })?;

        if tensor_value.device() != device {
            tensor_value = tensor_value.to(device).map_err(_convert_error)?;
        }

        let (input_cast, value_cast, _) =
            coerce_binary_operands(&self.inner, &tensor_value, BinaryOpKind::Add)
                .map_err(_convert_error)?;

        let input_tensor = match input_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let value_tensor = match value_cast {
            Cow::Borrowed(_) => tensor_value,
            Cow::Owned(tensor) => tensor,
        };

        let result = input_tensor
            .masked_fill(&mask_tensor, &value_tensor)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (other, axis=None))]
    pub fn cross(&self, other: &Bound<PyAny>, axis: Option<i32>) -> PyResult<Self> {
        let py = other.py();

        let maybe_tensor = if let Ok(tensor) = other.extract::<PyTensor>() {
            Some(tensor)
        } else if let Ok(attr) = other.getattr(intern!(py, "_tensor")) {
            attr.extract::<PyTensor>().ok()
        } else {
            None
        };

        let other_tensor = if let Some(tensor) = maybe_tensor {
            tensor
        } else {
            let dtype = self.inner.dtype();
            let device = self.inner.device();
            let converted = convert_python_data_to_tensor(other, dtype, device, false)?;
            PyTensor::from_tensor(converted)
        };

        cross_impl(self, &other_tensor, axis)
    }

    pub fn maximum(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Maximum)?;
        let result = lhs.maximum(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn minimum(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Minimum)?;
        let result = lhs.minimum(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn logaddexp(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.logaddexp(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn _coerce_binary_operands(
        &self,
        other: &PyTensor,
        op: &str,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let op_kind = match op {
            "__add__" | "add" | "logaddexp" => BinaryOpKind::Add,
            "__sub__" | "sub" => BinaryOpKind::Sub,
            "__mul__" | "mul" => BinaryOpKind::Mul,
            "__truediv__" | "div" => BinaryOpKind::Div,
            "maximum" => BinaryOpKind::Maximum,
            "minimum" => BinaryOpKind::Minimum,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported binary operation for dtype coercion: {op}"
                )));
            }
        };

        let (lhs_cast, rhs_cast, _) =
            coerce_binary_operands(self.tensor(), other.tensor(), op_kind)
                .map_err(_convert_error)?;

        let lhs_tensor = match lhs_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let rhs_tensor = match rhs_cast {
            Cow::Borrowed(_) => other.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };

        Ok((
            PyTensor::from_tensor(lhs_tensor),
            PyTensor::from_tensor(rhs_tensor),
        ))
    }

    // Comparison operations
    pub fn eq(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.eq_from_py(other)
    }

    pub fn ne(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ne_from_py(other)
    }

    pub fn lt(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.lt_from_py(other)
    }

    pub fn le(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.le_from_py(other)
    }

    pub fn gt(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.gt_from_py(other)
    }

    pub fn ge(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ge_from_py(other)
    }

    fn eq_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.eq(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn ne_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.ne(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn lt_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.lt(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn le_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.le(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn gt_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.gt(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn ge_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.ge(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    // Reduction operations
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.sum(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn logsumexp(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        match self.inner.logsumexp(dims, keepdim) {
            Ok(result) => Ok(Self::from_tensor(result)),
            Err(err @ MinitensorError::InvalidOperation { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn prod(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.prod(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.mean(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn all(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.all(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn any(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.any(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn cumsum(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumsum(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn cumprod(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumprod(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let values = Py::new(py, self.max_values(Some(dim), keepdim)?)?.into_any();
            let indices = Py::new(py, self.argmax(Some(dim), Some(keepdim))?)?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.max_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn min<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let values = Py::new(py, self.min_values(Some(dim), keepdim)?)?.into_any();
            let indices = Py::new(py, self.argmin(Some(dim), Some(keepdim))?)?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.min_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn median<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        let (values, indices) = self.median_with_indices(dim, keepdim)?;
        if dim.is_some() {
            let indices = indices.ok_or_else(|| {
                PyRuntimeError::new_err("median returned no indices for the requested dimension")
            })?;
            let values = Py::new(py, values)?.into_any();
            let indices = Py::new(py, indices)?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, values)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmin(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (k, dim=None, largest=true, sorted=true))]
    pub fn topk(
        &self,
        k: usize,
        dim: Option<isize>,
        largest: Option<bool>,
        sorted: Option<bool>,
    ) -> PyResult<(Self, Self)> {
        let largest = largest.unwrap_or(true);
        let sorted = sorted.unwrap_or(true);
        match self.inner.topk(k, dim, largest, sorted) {
            Ok((values, indices)) => Ok((Self::from_tensor(values), Self::from_tensor(indices))),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, descending=false, stable=false))]
    pub fn sort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<(Self, Self)> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.sort(dim, descending, stable) {
            Ok((values, indices)) => Ok((Self::from_tensor(values), Self::from_tensor(indices))),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, descending=false, stable=false))]
    pub fn argsort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<Self> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.argsort(dim, descending, stable) {
            Ok(indices) => Ok(Self::from_tensor(indices)),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn std(
        &self,
        dim: Option<isize>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .std(dim, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn var(
        &self,
        dim: Option<isize>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .var(dim, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    // Mathematical functions
    fn abs(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn sqrt(&self) -> PyResult<Self> {
        let result = self.inner.sqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn rsqrt(&self) -> PyResult<Self> {
        let result = self.inner.rsqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn pow(&self, exponent: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(exp_tensor) = exponent.extract::<PyTensor>() {
            let result = self.inner.pow(&exp_tensor.inner).map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(exp) = exponent.extract::<f64>() {
            let result = self.inner.powf(exp).map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        let exp_tensor = tensor_from_py_value(&self.inner, exponent)?;
        let result = self.inner.pow(&exp_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn exp(&self) -> PyResult<Self> {
        let result = self.inner.exp().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn log(&self) -> PyResult<Self> {
        let result = self.inner.log().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn log1p(&self) -> PyResult<Self> {
        let result = self.inner.log1p().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn expm1(&self) -> PyResult<Self> {
        let result = self.inner.expm1().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sin(&self) -> PyResult<Self> {
        let result = self.inner.sin().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn cos(&self) -> PyResult<Self> {
        let result = self.inner.cos().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn tan(&self) -> PyResult<Self> {
        let result = self.inner.tan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn clamp(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        let result = self.inner.clamp(min, max).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isnan(&self) -> PyResult<Self> {
        let result = self.inner.isnan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isinf(&self) -> PyResult<Self> {
        let result = self.inner.isinf().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isfinite(&self) -> PyResult<Self> {
        let result = self.inner.isfinite().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __pow__(&self, exponent: &Bound<PyAny>, _mod: Option<&Bound<PyAny>>) -> PyResult<Self> {
        self.pow(exponent)
    }

    fn __rpow__(&self, base: &Bound<PyAny>, _mod: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let base_tensor = tensor_from_py_value(&self.inner, base)?;
        let result = base_tensor.pow(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn relu(&self) -> PyResult<Self> {
        let result = self.inner.relu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn hardshrink(&self, lambd: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .hardshrink(lambd.unwrap_or(0.5))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self.inner.softmax(resolved_dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn log_softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self
            .inner
            .log_softmax(resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (normalized_shape, weight=None, bias=None, eps=1e-5))]
    pub fn layer_norm(
        &self,
        normalized_shape: Vec<usize>,
        weight: Option<&PyTensor>,
        bias: Option<&PyTensor>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        if normalized_shape.is_empty() {
            return Err(PyValueError::new_err(
                "layer_norm requires normalized_shape to contain at least one dimension",
            ));
        }

        let weight_inner = weight.map(|w| &w.inner);
        let bias_inner = bias.map(|b| &b.inner);
        let result = self
            .inner
            .layer_norm(
                &normalized_shape,
                weight_inner,
                bias_inner,
                eps.unwrap_or(1e-5),
            )
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn gelu(&self, approximate: Option<&str>) -> PyResult<Self> {
        let approx_mode = approximate.unwrap_or("none");
        let approximate = if approx_mode.eq_ignore_ascii_case("none") {
            false
        } else if approx_mode.eq_ignore_ascii_case("tanh") {
            true
        } else {
            return Err(PyValueError::new_err(
                "approximate must be 'none' or 'tanh' for gelu",
            ));
        };

        let result = self.inner.gelu(approximate).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sigmoid(&self) -> PyResult<Self> {
        let result = self.inner.sigmoid().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn softplus(&self, beta: Option<f64>, threshold: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .softplus(beta.unwrap_or(1.0), threshold.unwrap_or(20.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn elu(&self, alpha: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .elu(alpha.unwrap_or(1.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn selu(&self) -> PyResult<Self> {
        let result = self.inner.selu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn silu(&self) -> PyResult<Self> {
        let result = self.inner.silu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn softsign(&self) -> PyResult<Self> {
        let result = self.inner.softsign().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn tanh(&self) -> PyResult<Self> {
        let result = self.inner.tanh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    // NumPy conversion methods
    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, false)
    }

    fn numpy_copy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, true)
    }

    #[pyo3(signature = (dtype=None))]
    fn __array__(&self, py: Python, dtype: Option<&Bound<PyAny>>) -> PyResult<Py<PyAny>> {
        let array = self.numpy(py)?;
        if let Some(dtype_obj) = dtype {
            let array_bound = array.bind(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "copy"), false)?;
            let casted =
                array_bound.call_method(intern!(py, "astype"), (dtype_obj,), Some(&kwargs))?;
            Ok(casted.into())
        } else {
            Ok(array)
        }
    }

    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__(
        &self,
        py: Python,
        ufunc: &Bound<PyAny>,
        method: &str,
        inputs: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if method != "__call__" {
            return py_not_implemented(py);
        }

        if let Some(mapping) = kwargs {
            if let Some(out) = mapping.get_item("out")? {
                if !out.is_none() {
                    return py_not_implemented(py);
                }
            }
        }

        let mut operands: Vec<Tensor> = Vec::with_capacity(inputs.len());
        for value in inputs.iter() {
            match tensor_from_py_value(&self.inner, &value) {
                Ok(tensor) => operands.push(tensor),
                Err(_) => return py_not_implemented(py),
            }
        }

        let Some(name_obj) = ufunc.getattr(intern!(py, "__name__")).ok() else {
            return py_not_implemented(py);
        };
        let name = name_obj.str()?.to_str()?.to_ascii_lowercase();

        let result = match (name.as_str(), operands.len()) {
            ("add", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Add, |lhs, rhs| lhs.add(rhs))?
            }
            ("subtract", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Sub, |lhs, rhs| {
                engine::operations::arithmetic::sub(lhs, rhs)
            })?,
            ("multiply", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Mul, |lhs, rhs| {
                engine::operations::arithmetic::mul(lhs, rhs)
            })?,
            ("true_divide", 2) | ("divide", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Div, |lhs, rhs| {
                    engine::operations::arithmetic::div(lhs, rhs)
                })?
            }
            ("power", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Mul, |lhs, rhs| lhs.pow(rhs))?
            }
            ("maximum", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Maximum, |lhs, rhs| {
                lhs.maximum(rhs)
            })?,
            ("minimum", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Minimum, |lhs, rhs| {
                lhs.minimum(rhs)
            })?,
            ("negative", 1) => apply_unary_ufunc(&operands, |tensor| {
                engine::operations::arithmetic::neg(tensor)
            })?,
            ("absolute", 1) | ("abs", 1) => apply_unary_ufunc(&operands, |tensor| tensor.abs())?,
            ("exp", 1) => apply_unary_ufunc(&operands, |tensor| tensor.exp())?,
            ("log", 1) => apply_unary_ufunc(&operands, |tensor| tensor.log())?,
            ("sin", 1) => apply_unary_ufunc(&operands, |tensor| tensor.sin())?,
            ("cos", 1) => apply_unary_ufunc(&operands, |tensor| tensor.cos())?,
            ("tan", 1) => apply_unary_ufunc(&operands, |tensor| tensor.tan())?,
            ("sqrt", 1) => apply_unary_ufunc(&operands, |tensor| tensor.sqrt())?,
            _ => return py_not_implemented(py),
        };

        let py_tensor = Py::new(py, PyTensor::from_tensor(result))?;
        Ok(py_tensor.into_any())
    }

    fn tolist(&self) -> PyResult<Py<PyAny>> {
        if self.inner.ndim() == 0 {
            Python::attach(|py| convert_tensor_to_python_scalar(&self.inner, py))
        } else {
            Python::attach(|py| convert_tensor_to_python_list(&self.inner, py))
        }
    }

    fn item(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| convert_tensor_to_python_scalar(&self.inner, py))
    }

    // Comparison operations
    pub fn array_equal(&self, other: &PyTensor) -> PyResult<bool> {
        Ok(self.inner.array_equal(&other.inner))
    }

    pub fn allclose(
        &self,
        other: &PyTensor,
        rtol: Option<f64>,
        atol: Option<f64>,
    ) -> PyResult<bool> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);
        Ok(self.inner.allclose(&other.inner, rtol, atol))
    }

    // String representations
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={}, requires_grad={})",
            self.inner.shape().dims(),
            self.dtype(),
            self.device(),
            self.inner.requires_grad()
        )
    }

    fn __str__(&self) -> String {
        if self.inner.numel() <= 100 {
            match self.tolist() {
                Ok(data) => Python::attach(|py| format!("tensor({})", data.bind(py))),
                Err(_) => self.__repr__(),
            }
        } else {
            self.__repr__()
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.inner.ndim() == 0 {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object",
            ))
        } else {
            Ok(self.inner.shape().dims()[0])
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "The truth value of a tensor with more than one element is ambiguous",
            ));
        }

        match self.inner.dtype() {
            DataType::Float32 => {
                let data = self.inner.data().as_f32_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
                })?;
                Ok(data[0] != 0.0)
            }
            DataType::Float64 => {
                let data = self.inner.data().as_f64_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
                })?;
                Ok(data[0] != 0.0)
            }
            DataType::Int32 => {
                let data = self.inner.data().as_i32_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
                })?;
                Ok(data[0] != 0)
            }
            DataType::Int64 => {
                let data = self.inner.data().as_i64_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
                })?;
                Ok(data[0] != 0)
            }
            DataType::Bool => {
                let data = self.inner.data().as_bool_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
                })?;
                Ok(data[0])
            }
        }
    }

    fn __getitem__(&self, key: &Bound<PyAny>) -> PyResult<Self> {
        let indices = parse_indices(key, self.inner.shape().dims())?;
        let result = self.inner.index(&indices).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __setitem__(&mut self, key: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<()> {
        let indices = parse_indices(key, self.inner.shape().dims())?;
        let val_tensor = if let Ok(t) = value.extract::<PyTensor>() {
            t.inner
        } else {
            convert_python_data_to_tensor(value, self.inner.dtype(), self.inner.device(), false)?
        };
        self.inner
            .index_assign(&indices, &val_tensor)
            .map_err(_convert_error)?;
        Ok(())
    }

    // Static tensor creation methods
    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn empty(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::empty(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn zeros(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::zeros(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn ones(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::ones(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn rand(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, false)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn randn(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, true)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, dtype=None, device=None, requires_grad=false))]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let m = m.unwrap_or(n);
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_eye_tensor(n, m, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype=None, device=None, requires_grad=false))]
    pub fn full(
        shape: &Bound<PyAny>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let dims = parse_shape_like(shape, "shape")?;
        let tensor = create_full_tensor(dims, fill_value, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (start, end=None, step=1.0, dtype=None, device=None, requires_grad=false))]
    fn arange(
        start: f64,
        end: Option<f64>,
        step: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let (start, end) = match end {
            Some(value) => (start, value),
            None => (0.0, start),
        };

        let tensor = create_arange_tensor(start, end, step, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, steps, dtype=None, device=None, requires_grad=false))]
    fn linspace(
        start: f64,
        end: f64,
        steps: usize,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        if steps == 0 {
            return Err(PyValueError::new_err("steps must be greater than zero"));
        }

        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_linspace_tensor(start, end, steps, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, steps, base=None, dtype=None, device=None, requires_grad=false))]
    fn logspace(
        start: f64,
        end: f64,
        steps: usize,
        base: Option<f64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        if steps == 0 {
            return Err(PyValueError::new_err("steps must be greater than zero"));
        }

        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);
        let base = base.unwrap_or(10.0);

        let tensor = create_logspace_tensor(start, end, steps, base, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        let tensor = convert_numpy_to_tensor(array, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy_shared(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        // For now, same as from_numpy - true zero-copy would require more complex memory management
        Self::from_numpy(array, requires_grad)
    }

    /// Concatenate tensors along an axis
    #[staticmethod]
    pub fn concatenate(tensors: &Bound<PyList>, _axis: Option<isize>) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty list of tensors",
            ));
        }

        let axis = _axis.unwrap_or(0);

        let tensor_vec: Vec<Tensor> = tensors
            .iter()
            .map(|obj| PyTensor::from_python_value(&obj).map(|t| t.inner.clone()))
            .collect::<PyResult<_>>()?;

        let tensor_refs: Vec<&Tensor> = tensor_vec.iter().collect();
        let result = engine::operations::shape_ops::concatenate(&tensor_refs, axis)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Stack tensors along a new axis
    #[staticmethod]
    pub fn stack(tensors: &Bound<PyList>, _axis: Option<isize>) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot stack empty list of tensors",
            ));
        }

        let axis = _axis.unwrap_or(0);

        let unsqueezed: Vec<Tensor> = tensors
            .iter()
            .map(|obj| {
                let t = PyTensor::from_python_value(&obj)?;
                t.inner.unsqueeze(axis as isize).map_err(_convert_error)
            })
            .collect::<PyResult<_>>()?;

        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        let result =
            engine::operations::shape_ops::concatenate(&refs, axis).map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Select elements along a dimension using integer indices
    pub fn index_select(&self, dim: isize, indices: &Bound<PyList>) -> PyResult<PyTensor> {
        let idx_vec: Vec<usize> = indices.extract()?;
        let result = engine::operations::shape_ops::index_select(&self.inner, dim, &idx_vec)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Gather elements along a dimension using an index tensor
    pub fn gather(&self, dim: isize, index: &PyTensor) -> PyResult<PyTensor> {
        let result = engine::operations::shape_ops::gather(&self.inner, dim, &index.inner)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Split tensor into multiple sub-tensors of equal size (``chunk``)
    #[pyo3(signature = (sections, dim=0))]
    pub fn chunk(&self, sections: usize, dim: isize) -> PyResult<Vec<PyTensor>> {
        if sections <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sections must be greater than zero",
            ));
        }

        let ndim = self.inner.ndim() as isize;
        let axis = if dim < 0 { dim + ndim } else { dim };
        if axis < 0 || axis >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} out of range",
                axis
            )));
        }

        let dim_size = self.inner.shape().dims()[axis as usize];
        if dim_size % sections != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Tensor cannot be evenly split along the given axis",
            ));
        }

        let chunk_size = dim_size / sections;
        let section_vec = vec![chunk_size as usize; sections as usize];
        self.split_with_sections(section_vec, axis as usize)
    }

    /// Split tensor by chunk size or explicit sections along an axis
    #[pyo3(signature = (split_size_or_sections, dim=0))]
    pub fn split(
        &self,
        split_size_or_sections: &Bound<PyAny>,
        dim: Option<isize>,
    ) -> PyResult<Vec<PyTensor>> {
        let dim = dim.unwrap_or(0);
        let ndim = self.inner.ndim() as isize;
        let dim = if dim < 0 { dim + ndim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} out of range",
                dim
            )));
        }
        let axis = dim as usize;
        let dim_size = self.inner.shape().dims()[axis];

        let mut sections: Vec<usize> = Vec::new();

        if let Ok(split_size) = split_size_or_sections.extract::<usize>() {
            if split_size == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split_size must be greater than zero",
                ));
            }
            let mut remaining = dim_size;
            while remaining > 0 {
                let chunk = split_size.min(remaining);
                sections.push(chunk);
                remaining -= chunk;
            }
        } else if let Ok(list) = split_size_or_sections.downcast::<PyList>() {
            for obj in list.iter() {
                let size: usize = obj.extract()?;
                if size == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "section size must be greater than zero",
                    ));
                }
                sections.push(size);
            }
            let total: usize = sections.iter().sum();
            if total != dim_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split sizes do not sum to dimension size",
                ));
            }
        } else if let Ok(tuple) = split_size_or_sections.downcast::<PyTuple>() {
            for obj in tuple.iter() {
                let size: usize = obj.extract()?;
                if size == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "section size must be greater than zero",
                    ));
                }
                sections.push(size);
            }
            let total: usize = sections.iter().sum();
            if total != dim_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split sizes do not sum to dimension size",
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "split_size_or_sections must be int or sequence",
            ));
        }

        self.split_with_sections(sections, axis)
    }

    fn split_with_sections(&self, sections: Vec<usize>, axis: usize) -> PyResult<Vec<PyTensor>> {
        let mut outputs = Vec::with_capacity(sections.len());
        let mut start = 0;
        for size in sections {
            let end = start + size;
            let slice =
                engine::operations::shape_ops::slice(&self.inner, axis as isize, start, end, 1)
                    .map_err(_convert_error)?;
            outputs.push(PyTensor::from_tensor(slice));
            start = end;
        }
        Ok(outputs)
    }
}

fn convert_dimension(value: isize, arg_name: &str) -> PyResult<usize> {
    if value < 0 {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must contain non-negative integers",
        )));
    }

    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{arg_name} value is too large for this platform",))
    })
}

fn convert_dimensions(values: Vec<isize>, arg_name: &str) -> PyResult<Vec<usize>> {
    let mut dims = Vec::with_capacity(values.len());
    for value in values {
        dims.push(convert_dimension(value, arg_name)?);
    }
    Ok(dims)
}

fn normalize_variadic_isize_args(tuple: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<isize>> {
    if tuple.is_empty() {
        return Ok(Vec::new());
    }

    if tuple.len() == 1 {
        let first = tuple.get_item(0)?;

        if let Ok(nested) = first.downcast::<PyTuple>() {
            return normalize_variadic_isize_args(&nested, arg_name);
        }

        if let Ok(list) = first.downcast::<PyList>() {
            let mut dims = Vec::with_capacity(list.len());
            for item in list.iter() {
                dims.push(item.extract::<isize>()?);
            }
            return Ok(dims);
        }

        if let Ok(shape_sequence) = first.extract::<ShapeSequence>() {
            return convert_usize_list_to_isize(shape_sequence.to_list(), arg_name);
        }

        if let Ok(values) = first.extract::<Vec<isize>>() {
            return Ok(values);
        }

        if let Ok(values) = first.extract::<Vec<usize>>() {
            return convert_usize_list_to_isize(values, arg_name);
        }

        if let Ok(value) = first.extract::<isize>() {
            return Ok(vec![value]);
        }

        if let Ok(value) = first.extract::<usize>() {
            return Ok(vec![convert_usize_to_isize(value, arg_name)?]);
        }
    }

    let mut dims = Vec::with_capacity(tuple.len());
    for item in tuple.iter() {
        dims.push(item.extract::<isize>()?);
    }
    Ok(dims)
}

fn convert_usize_list_to_isize(values: Vec<usize>, arg_name: &str) -> PyResult<Vec<isize>> {
    let mut converted = Vec::with_capacity(values.len());
    for value in values {
        converted.push(convert_usize_to_isize(value, arg_name)?);
    }
    Ok(converted)
}

fn convert_usize_to_isize(value: usize, arg_name: &str) -> PyResult<isize> {
    isize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "{arg_name} dimension {value} is too large for this platform"
        ))
    })
}

fn parse_shape_tuple(shape: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<usize>> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }

    if shape.len() == 1 {
        let first = shape.get_item(0)?;
        if let Ok(tuple) = first.downcast::<PyTuple>() {
            return parse_shape_tuple(&tuple, arg_name);
        }
        if let Ok(list) = first.downcast::<PyList>() {
            let mut dims = Vec::with_capacity(list.len());
            for item in list.iter() {
                let value: isize = item.extract()?;
                dims.push(convert_dimension(value, arg_name)?);
            }
            return Ok(dims);
        }
        if let Ok(shape_seq) = first.extract::<ShapeSequence>() {
            return Ok(shape_seq.to_list());
        }
        if let Ok(values) = first.extract::<Vec<isize>>() {
            return convert_dimensions(values, arg_name);
        }
        if let Ok(value) = first.extract::<isize>() {
            return Ok(vec![convert_dimension(value, arg_name)?]);
        }
    }

    let mut dims = Vec::with_capacity(shape.len());
    for item in shape.iter() {
        let value: isize = item.extract()?;
        dims.push(convert_dimension(value, arg_name)?);
    }
    Ok(dims)
}

fn parse_shape_like(obj: &Bound<PyAny>, arg_name: &str) -> PyResult<Vec<usize>> {
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return parse_shape_tuple(&tuple, arg_name);
    }

    if let Ok(list) = obj.downcast::<PyList>() {
        let mut dims = Vec::with_capacity(list.len());
        for item in list.iter() {
            let value: isize = item.extract()?;
            dims.push(convert_dimension(value, arg_name)?);
        }
        return Ok(dims);
    }

    if let Ok(shape_seq) = obj.extract::<ShapeSequence>() {
        return Ok(shape_seq.to_list());
    }

    if let Ok(values) = obj.extract::<Vec<isize>>() {
        return convert_dimensions(values, arg_name);
    }

    if let Ok(value) = obj.extract::<isize>() {
        return Ok(vec![convert_dimension(value, arg_name)?]);
    }

    Err(PyTypeError::new_err(format!(
        "{arg_name} must be an int or sequence of ints",
    )))
}

fn normalize_roll_shifts(shifts: &Bound<PyAny>) -> PyResult<Vec<isize>> {
    normalize_required_axes(shifts, "shifts")
}

fn normalize_required_axes<'py>(dim: &'py Bound<'py, PyAny>, name: &str) -> PyResult<Vec<isize>> {
    match normalize_optional_axes(Some(dim))? {
        Some(values) => Ok(values),
        None => Err(PyTypeError::new_err(format!(
            "{} must be an int or a sequence of ints",
            name
        ))),
    }
}

fn normalize_optional_axes(dim: Option<&Bound<PyAny>>) -> PyResult<Option<Vec<isize>>> {
    let Some(obj) = dim else {
        return Ok(None);
    };

    if obj.is_none() {
        return Ok(None);
    }

    if is_bool_axis(obj)? {
        return Err(PyTypeError::new_err(
            "dim must be an int or a sequence of ints",
        ));
    }

    if let Ok(value) = obj.extract::<isize>() {
        return Ok(Some(vec![value]));
    }

    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "dim must be an int or a sequence of ints",
        ));
    }

    if let Ok(sequence) = obj.downcast::<PySequence>() {
        let length = sequence.len()? as usize;
        let mut axes = Vec::with_capacity(length);
        for index in 0..length {
            let item = sequence.get_item(index)?;
            if is_bool_axis(&item)? {
                return Err(PyTypeError::new_err(
                    "dim must be an int or a sequence of ints",
                ));
            }
            let value: isize = item.extract()?;
            axes.push(value);
        }
        return Ok(Some(axes));
    }

    Err(PyTypeError::new_err(
        "dim must be an int or a sequence of ints",
    ))
}

fn is_bool_axis(obj: &Bound<PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(true);
    }

    static NUMPY_BOOL_TYPE: OnceCell<Py<PyAny>> = OnceCell::new();
    let py = obj.py();
    if let Ok(numpy_bool) = NUMPY_BOOL_TYPE.get_or_try_init(|| -> PyResult<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let bool_obj = numpy.getattr("bool_")?;
        Ok(bool_obj.unbind())
    }) {
        if obj.is_instance(&numpy_bool.bind(py))? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn normalize_repeat_spec(repeats: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if repeats.is_instance_of::<PyString>() {
        return Ok(vec![extract_repeat_element(repeats)?]);
    }

    if let Ok(sequence) = repeats.extract::<Vec<i64>>() {
        let mut values = Vec::with_capacity(sequence.len());
        for repeat in sequence {
            if repeat < 0 {
                return Err(PyValueError::new_err(
                    "repeat expects non-negative integers",
                ));
            }
            values.push(repeat as usize);
        }
        return Ok(values);
    }

    Ok(vec![extract_repeat_element(repeats)?])
}

fn extract_repeat_element(value: &Bound<PyAny>) -> PyResult<usize> {
    let repeat: i64 = value.extract()?;
    if repeat < 0 {
        Err(PyValueError::new_err(
            "repeat expects non-negative integers",
        ))
    } else {
        Ok(repeat as usize)
    }
}

fn convert_python_data_to_tensor(
    data: &Bound<PyAny>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    // First try NumPy array conversion for any supported dtype
    if let Ok(numpy_module) = PyModule::import(data.py(), "numpy") {
        if let Ok(ndarray_type) = numpy_module.getattr("ndarray") {
            if data.is_instance(&ndarray_type)? {
                let maybe_tensor = panic::catch_unwind(AssertUnwindSafe(|| {
                    convert_numpy_to_tensor(data, requires_grad)
                }));

                match maybe_tensor {
                    Ok(Ok(tensor)) => {
                        let tensor = if tensor.dtype() != dtype {
                            tensor.astype(dtype).map_err(_convert_error)?
                        } else {
                            tensor
                        };
                        return Ok(tensor);
                    }
                    Ok(Err(err)) => {
                        return Err(err);
                    }
                    Err(_) => {
                        // Fall back to the slower Python list conversion path
                        // when the NumPy capsule isn't available.
                    }
                }
            }
        }
    }

    // Handle Python lists and tuples by flattening values into scalar variants
    if let Ok(list) = data.downcast::<PyList>() {
        let (shape, flat_data) = flatten_python_data(list)?;
        let (base_tensor, base_dtype) =
            tensor_from_flat_scalars(shape, flat_data, device, requires_grad)?;

        if base_dtype == dtype {
            return Ok(base_tensor);
        }

        return base_tensor.astype(dtype).map_err(_convert_error);
    }

    if let Ok(tuple) = data.downcast::<PyTuple>() {
        let list = tuple.to_list();
        return convert_python_data_to_tensor(list.as_any(), dtype, device, requires_grad);
    }

    // Handle scalars
    if let Ok(value_bool) = data.extract::<bool>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_bool(vec![value_bool], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Bool, device, requires_grad);
        if dtype != DataType::Bool {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(value_int) = data.extract::<i64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_i64(vec![value_int], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Int64, device, requires_grad);
        if dtype != DataType::Int64 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(value_float) = data.extract::<f64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_f64(vec![value_float], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
        if dtype != DataType::Float64 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    let float_name = intern!(data.py(), "__float__");
    if data.hasattr(float_name)? {
        let method = data.getattr(float_name)?;
        if method.is_callable() {
            let float_obj = method.call0()?;
            let val = float_obj.extract::<f64>()?;
            let shape = Shape::new(vec![]);
            let base_data = Arc::new(TensorData::from_vec_f64(vec![val], device));
            let mut tensor =
                Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
            if dtype != DataType::Float64 {
                tensor = tensor.astype(dtype).map_err(_convert_error)?;
            }
            return Ok(tensor);
        }
    }

    Err(PyErr::new::<PyTypeError, _>(
        "Unsupported data type for tensor creation",
    ))
}

fn apply_binary_ufunc<F>(operands: &[Tensor], kind: BinaryOpKind, op: F) -> PyResult<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor, MinitensorError>,
{
    if operands.len() != 2 {
        return Err(PyValueError::new_err(
            "Binary ufuncs require exactly two operands",
        ));
    }

    let (lhs_cast, rhs_cast, _) =
        coerce_binary_operands(&operands[0], &operands[1], kind).map_err(_convert_error)?;

    let lhs_tensor = match lhs_cast {
        Cow::Borrowed(tensor) => tensor.clone(),
        Cow::Owned(tensor) => tensor,
    };
    let rhs_tensor = match rhs_cast {
        Cow::Borrowed(tensor) => tensor.clone(),
        Cow::Owned(tensor) => tensor,
    };

    op(&lhs_tensor, &rhs_tensor).map_err(_convert_error)
}

fn apply_unary_ufunc<F>(operands: &[Tensor], op: F) -> PyResult<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor, MinitensorError>,
{
    if operands.len() != 1 {
        return Err(PyValueError::new_err(
            "Unary ufuncs require exactly one operand",
        ));
    }

    let tensor = operands[0].clone();
    op(&tensor).map_err(_convert_error)
}

fn py_not_implemented(py: Python) -> PyResult<Py<PyAny>> {
    unsafe { Ok(Py::from_borrowed_ptr(py, pyo3::ffi::Py_NotImplemented())) }
}

fn parse_dtype_like(value: &Bound<PyAny>) -> PyResult<DataType> {
    if let Ok(name) = value.extract::<String>() {
        dtype::parse_dtype(&name)
    } else {
        Err(PyTypeError::new_err(
            "dtype must be specified as a string such as 'float32'",
        ))
    }
}

fn parse_device_like(value: &Bound<PyAny>) -> PyResult<Device> {
    if let Ok(device) = value.extract::<PyDevice>() {
        return Ok(device.device());
    }

    if let Ok(spec) = value.extract::<String>() {
        return Device::from_str(&spec).map_err(|err| {
            PyValueError::new_err(format!("Unsupported device specification '{spec}': {err}"))
        });
    }

    Err(PyTypeError::new_err(
        "device must be specified as a Device object or string like 'cpu' or 'cuda:0'",
    ))
}

fn ensure_backward_gradient_compatible(reference: &Tensor, gradient: &mut Tensor) -> PyResult<()> {
    let expected_shape = reference.shape().dims();
    let actual_shape = gradient.shape().dims();
    if expected_shape != actual_shape {
        return Err(PyRuntimeError::new_err(format!(
            "backward() expected gradient tensor with shape {:?}, but got {:?}",
            expected_shape, actual_shape
        )));
    }

    if gradient.device() != reference.device() {
        *gradient = gradient.to(reference.device()).map_err(_convert_error)?;
    }

    if gradient.dtype() != reference.dtype() {
        *gradient = gradient.astype(reference.dtype()).map_err(_convert_error)?;
    }

    if gradient.requires_grad() {
        *gradient = gradient.detach();
    }

    Ok(())
}

fn tensor_from_py_value(reference: &Tensor, value: &Bound<PyAny>) -> PyResult<Tensor> {
    if let Ok(py_tensor) = value.extract::<PyTensor>() {
        return Ok(py_tensor.inner.clone());
    }

    if let Ok(inner) = value.getattr("_tensor") {
        if let Ok(py_tensor) = inner.extract::<PyTensor>() {
            return Ok(py_tensor.inner.clone());
        }
    }

    if let Ok(numpy_module) = PyModule::import(value.py(), "numpy") {
        if let Ok(ndarray_type) = numpy_module.getattr("ndarray") {
            if value.is_instance(&ndarray_type)? {
                if let Ok(dtype_obj) = value.getattr("dtype") {
                    let dtype_str = dtype_obj.str()?.to_str()?.to_ascii_lowercase();
                    if let Ok(array_dtype) = dtype::parse_dtype(&dtype_str) {
                        return convert_python_data_to_tensor(
                            value,
                            array_dtype,
                            reference.device(),
                            false,
                        );
                    }
                }
                return convert_python_data_to_tensor(
                    value,
                    reference.dtype(),
                    reference.device(),
                    false,
                );
            }
        }
    }

    if let Ok(py_tensor) = PyTensor::from_python_value(value) {
        let mut tensor = py_tensor.inner;
        if tensor.device() != reference.device() {
            tensor = tensor.to(reference.device()).map_err(_convert_error)?;
        }

        let target_dtype = dtype::resolve_scalar_dtype(value, reference.dtype())
            .ok()
            .or_else(|| infer_python_value_dtype(value))
            .unwrap_or(reference.dtype());

        if tensor.dtype() != target_dtype {
            tensor = tensor.astype(target_dtype).map_err(_convert_error)?;
        }

        if target_dtype != reference.dtype() {
            return Ok(tensor);
        }

        if tensor.dtype() != reference.dtype() {
            tensor = tensor.astype(reference.dtype()).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    let index_name = intern!(value.py(), "__index__");
    if value.hasattr(index_name)? {
        let method = value.getattr(index_name)?;
        if method.is_callable() {
            let result = method.call0()?;
            if result.is_instance_of::<PyInt>() {
                let dtype = match dtype::resolve_scalar_dtype(value, reference.dtype()) {
                    Ok(dt) => dt,
                    Err(_) => reference.dtype(),
                };
                return convert_python_data_to_tensor(
                    result.as_any(),
                    dtype,
                    reference.device(),
                    false,
                );
            }
        }
    }

    let dtype = match dtype::resolve_scalar_dtype(value, reference.dtype()) {
        Ok(dt) => dt,
        Err(_) => infer_python_value_dtype(value).unwrap_or(reference.dtype()),
    };
    convert_python_data_to_tensor(value, dtype, reference.device(), false)
}

fn tensor_bool_from_py(value: &Bound<PyAny>, device: Device) -> PyResult<Tensor> {
    if let Ok(py_tensor) = value.extract::<PyTensor>() {
        let mut tensor = py_tensor.inner.clone();
        if tensor.dtype() != DataType::Bool {
            return Err(PyTypeError::new_err("mask must be a bool tensor"));
        }
        if tensor.device() != device {
            tensor = tensor.to(device).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(inner) = value.getattr("_tensor") {
        if let Ok(py_tensor) = inner.extract::<PyTensor>() {
            let mut tensor = py_tensor.inner.clone();
            if tensor.dtype() != DataType::Bool {
                return Err(PyTypeError::new_err("mask must be a bool tensor"));
            }
            if tensor.device() != device {
                tensor = tensor.to(device).map_err(_convert_error)?;
            }
            return Ok(tensor);
        }
    }

    if let Ok(value_bool) = value.extract::<bool>() {
        let data = Arc::new(TensorData::from_vec_bool(vec![value_bool], device));
        return Ok(Tensor::new(
            data,
            Shape::new(vec![]),
            DataType::Bool,
            device,
            false,
        ));
    }

    let mut tensor = convert_python_data_to_tensor(value, DataType::Bool, device, false)?;
    if tensor.dtype() != DataType::Bool {
        tensor = tensor.astype(DataType::Bool).map_err(_convert_error)?;
    }
    if tensor.device() != device {
        tensor = tensor.to(device).map_err(_convert_error)?;
    }
    Ok(tensor)
}

fn promote_dtypes(a: DataType, b: DataType) -> DataType {
    use DataType::*;

    if a == b {
        return a;
    }

    match (a, b) {
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) => Float32,
        (Int64, _) | (_, Int64) => Int64,
        (Int32, _) | (_, Int32) => Int32,
        _ => Bool,
    }
}

fn infer_python_value_dtype(value: &Bound<PyAny>) -> Option<DataType> {
    if let Ok(py_tensor) = value.extract::<PyTensor>() {
        return Some(py_tensor.inner.dtype());
    }

    if let Ok(inner) = value.getattr("_tensor") {
        if let Ok(py_tensor) = inner.extract::<PyTensor>() {
            return Some(py_tensor.inner.dtype());
        }
    }

    if value.extract::<bool>().is_ok() {
        return Some(DataType::Bool);
    }

    if value.extract::<i64>().is_ok() {
        return Some(DataType::Int64);
    }

    if value.extract::<f64>().is_ok() {
        return Some(dtype::default_dtype());
    }

    if let Ok(numpy_module) = PyModule::import(value.py(), "numpy") {
        if let Ok(ndarray_type) = numpy_module.getattr("ndarray") {
            if let Ok(true) = value.is_instance(&ndarray_type) {
                if let Ok(dtype_obj) = value.getattr("dtype") {
                    if let Ok(dtype_str) = dtype_obj.str() {
                        if let Ok(dtype) =
                            dtype::parse_dtype(&dtype_str.to_str().ok()?.to_ascii_lowercase())
                        {
                            return Some(dtype);
                        }
                    }
                }
            }
        }
    }

    if let Ok(list) = value.downcast::<PyList>() {
        return infer_sequence_dtype(list.iter());
    }

    if let Ok(tuple) = value.downcast::<PyTuple>() {
        return infer_sequence_dtype(tuple.iter());
    }

    None
}

fn infer_sequence_dtype<'py, I>(iter: I) -> Option<DataType>
where
    I: Iterator<Item = Bound<'py, PyAny>>,
{
    let mut dtype: Option<DataType> = None;
    for item in iter {
        let item_dtype = infer_python_value_dtype(&item)?;
        dtype = Some(match dtype {
            Some(current) => promote_dtypes(current, item_dtype),
            None => item_dtype,
        });
    }
    dtype
}

fn prepare_binary_operands_from_py(
    reference: &Tensor,
    other: &Bound<PyAny>,
    reverse: bool,
    kind: BinaryOpKind,
) -> PyResult<(Tensor, Tensor)> {
    let lhs_input = if reverse {
        tensor_from_py_value(reference, other)?
    } else {
        reference.clone()
    };

    let rhs_input = if reverse {
        reference.clone()
    } else {
        tensor_from_py_value(reference, other)?
    };

    let (lhs_cast, rhs_cast, _) =
        coerce_binary_operands(&lhs_input, &rhs_input, kind).map_err(_convert_error)?;
    let lhs_tensor = match lhs_cast {
        Cow::Borrowed(_) => lhs_input.clone(),
        Cow::Owned(tensor) => tensor,
    };
    let rhs_tensor = match rhs_cast {
        Cow::Borrowed(_) => rhs_input.clone(),
        Cow::Owned(tensor) => tensor,
    };

    Ok((lhs_tensor, rhs_tensor))
}

fn flatten_python_data(list: &Bound<PyList>) -> PyResult<(Vec<usize>, Vec<ScalarValue>)> {
    let mut shape = vec![list.len()];
    let mut flat_data = vec![];

    fn process_nested(
        item: &Bound<PyAny>,
        depth: usize,
        shape: &mut Vec<usize>,
        flat_data: &mut Vec<ScalarValue>,
    ) -> PyResult<()> {
        if let Ok(nested_list) = item.downcast::<PyList>() {
            let length = nested_list.len();
            if depth >= shape.len() {
                shape.push(length);
            } else if shape[depth] != length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Inconsistent nested sequence lengths",
                ));
            }
            for nested_item in nested_list.iter() {
                process_nested(&nested_item, depth + 1, shape, flat_data)?;
            }
            return Ok(());
        }

        if let Ok(nested_tuple) = item.downcast::<PyTuple>() {
            let list = nested_tuple.to_list();
            let length = list.len();
            if depth >= shape.len() {
                shape.push(length);
            } else if shape[depth] != length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Inconsistent nested sequence lengths",
                ));
            }
            for nested_item in list.iter() {
                process_nested(&nested_item, depth + 1, shape, flat_data)?;
            }
            return Ok(());
        }

        if let Ok(value_bool) = item.extract::<bool>() {
            flat_data.push(ScalarValue::Bool(value_bool));
            return Ok(());
        }

        if let Ok(value_int) = item.extract::<i64>() {
            flat_data.push(ScalarValue::Int(value_int));
            return Ok(());
        }

        let index_name = intern!(item.py(), "__index__");
        if item.hasattr(index_name)? {
            let method = item.getattr(index_name)?;
            if method.is_callable() {
                let result = method.call0()?;
                if result.is_instance_of::<PyInt>() {
                    let value = result.extract::<i64>()?;
                    flat_data.push(ScalarValue::Int(value));
                    return Ok(());
                }
            }
        }

        if let Ok(value_float) = item.extract::<f64>() {
            flat_data.push(ScalarValue::Float(value_float));
            return Ok(());
        }

        let float_name = intern!(item.py(), "__float__");
        if item.hasattr(float_name)? {
            let method = item.getattr(float_name)?;
            if method.is_callable() {
                let float_obj = method.call0()?;
                let value = float_obj.extract::<f64>()?;
                flat_data.push(ScalarValue::Float(value));
                return Ok(());
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported scalar type in nested sequence",
        ))
    }

    for item in list.iter() {
        process_nested(&item, 1, &mut shape, &mut flat_data)?;
    }

    Ok((shape, flat_data))
}

#[derive(Clone, Copy)]
enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl ScalarValue {
    fn kind(&self) -> ScalarKind {
        match self {
            ScalarValue::Bool(_) => ScalarKind::Bool,
            ScalarValue::Int(_) => ScalarKind::Int,
            ScalarValue::Float(_) => ScalarKind::Float,
        }
    }

    fn to_bool(self) -> bool {
        match self {
            ScalarValue::Bool(value) => value,
            ScalarValue::Int(value) => value != 0,
            ScalarValue::Float(value) => value != 0.0,
        }
    }

    fn to_i64(self) -> i64 {
        match self {
            ScalarValue::Bool(value) => value as i64,
            ScalarValue::Int(value) => value,
            ScalarValue::Float(value) => value as i64,
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            ScalarValue::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
            ScalarValue::Int(value) => value as f64,
            ScalarValue::Float(value) => value,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Bool,
    Int,
    Float,
}

impl ScalarKind {
    fn combine(self, other: ScalarKind) -> ScalarKind {
        use ScalarKind::*;
        match (self, other) {
            (Float, _) | (_, Float) => Float,
            (Int, _) | (_, Int) => Int,
            _ => Bool,
        }
    }
}

fn tensor_from_flat_scalars(
    shape: Vec<usize>,
    values: Vec<ScalarValue>,
    device: Device,
    requires_grad: bool,
) -> PyResult<(Tensor, DataType)> {
    let mut kind = ScalarKind::Bool;
    for value in &values {
        kind = kind.combine(value.kind());
    }

    let tensor = match kind {
        ScalarKind::Bool => {
            let data: Vec<bool> = values.into_iter().map(ScalarValue::to_bool).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_bool(data, device)),
                Shape::new(shape),
                DataType::Bool,
                device,
                requires_grad,
            )
        }
        ScalarKind::Int => {
            let data: Vec<i64> = values.into_iter().map(ScalarValue::to_i64).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_i64(data, device)),
                Shape::new(shape),
                DataType::Int64,
                device,
                requires_grad,
            )
        }
        ScalarKind::Float => {
            let data: Vec<f64> = values.into_iter().map(ScalarValue::to_f64).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_f64(data, device)),
                Shape::new(shape),
                DataType::Float64,
                device,
                requires_grad,
            )
        }
    };

    let dtype = tensor.dtype();
    Ok((tensor, dtype))
}

fn parse_index(item: &Bound<PyAny>, dim_size: usize) -> PyResult<TensorIndex> {
    if let Ok(i) = item.extract::<isize>() {
        let mut idx = i;
        if idx < 0 {
            idx += dim_size as isize;
        }
        if idx < 0 || idx >= dim_size as isize {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        Ok(TensorIndex::Index(idx as usize))
    } else if let Ok(slice) = item.downcast::<PySlice>() {
        use std::convert::TryInto;

        let dim_size_isize: isize = dim_size
            .try_into()
            .map_err(|_| PyValueError::new_err("dim_size too large"))?;
        let indices = slice.indices(dim_size_isize)?;
        if indices.step <= 0 {
            return Err(PyIndexError::new_err("slice step must be positive"));
        }
        Ok(TensorIndex::Slice {
            start: indices.start.max(0) as usize,
            end: indices.stop.max(0) as usize,
            step: indices.step as usize,
        })
    } else if item.is_none() {
        Ok(TensorIndex::Slice {
            start: 0,
            end: dim_size,
            step: 1,
        })
    } else {
        Err(PyTypeError::new_err("Invalid index type"))
    }
}

fn parse_indices(key: &Bound<PyAny>, shape: &[usize]) -> PyResult<Vec<TensorIndex>> {
    if let Ok(tup) = key.downcast::<PyTuple>() {
        if tup.len() > shape.len() {
            return Err(PyIndexError::new_err("Too many indices"));
        }
        let mut result = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if i < tup.len() {
                result.push(parse_index(&tup.get_item(i)?, *dim)?);
            } else {
                result.push(TensorIndex::Slice {
                    start: 0,
                    end: *dim,
                    step: 1,
                });
            }
        }
        Ok(result)
    } else {
        let mut result = vec![parse_index(key, shape[0])?];
        for dim in &shape[1..] {
            result.push(TensorIndex::Slice {
                start: 0,
                end: *dim,
                step: 1,
            });
        }
        Ok(result)
    }
}

fn convert_numpy_to_tensor(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Tensor> {
    if let Ok(array_f32) = array.downcast::<PyArrayDyn<f32>>() {
        let readonly = array_f32.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<f32> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Float32,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_f64) = array.downcast::<PyArrayDyn<f64>>() {
        let readonly = array_f64.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<f64> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Float64,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_i32) = array.downcast::<PyArrayDyn<i32>>() {
        let readonly = array_i32.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<i32> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Int32,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Int32,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_i64) = array.downcast::<PyArrayDyn<i64>>() {
        let readonly = array_i64.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<i64> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Int64,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Int64,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_bool) = array.downcast::<PyArrayDyn<bool>>() {
        let readonly = array_bool.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<bool> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Bool,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Bool,
            Device::cpu(),
            requires_grad,
        ))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported NumPy array type",
        ))
    }
}

fn convert_tensor_to_numpy(tensor: &Tensor, py: Python, _force_copy: bool) -> PyResult<Py<PyAny>> {
    if tensor.device() != Device::cpu() {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Cannot convert GPU tensor to NumPy array. Use .cpu() first.",
        ));
    }

    let shape = tensor.shape().dims();
    let strides = tensor.strides().as_slice();
    let numel: usize = shape.iter().product();

    macro_rules! to_numpy {
        ($slice:expr, $ty:ty) => {{
            let data = $slice.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get tensor data")
            })?;
            let mut out = Vec::<$ty>::with_capacity(numel);
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..numel {
                let mut offset = 0usize;
                for (idx, stride) in indices.iter().zip(strides) {
                    offset += idx * stride;
                }
                out.push(data[offset]);
                for axis in (0..indices.len()).rev() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
            let array = PyArray::from_vec(py, out).reshape(shape)?;
            Ok(array.into_any().unbind())
        }};
    }

    let array: PyResult<Py<PyAny>> = match tensor.dtype() {
        DataType::Float32 => to_numpy!(tensor.data().as_f32_slice(), f32),
        DataType::Float64 => to_numpy!(tensor.data().as_f64_slice(), f64),
        DataType::Int32 => to_numpy!(tensor.data().as_i32_slice(), i32),
        DataType::Int64 => to_numpy!(tensor.data().as_i64_slice(), i64),
        DataType::Bool => to_numpy!(tensor.data().as_bool_slice(), bool),
    };

    Ok(array?)
}

fn convert_tensor_to_python_list(tensor: &Tensor, py: Python) -> PyResult<Py<PyAny>> {
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor.data().as_f32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Float64 => {
            let data = tensor.data().as_f64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Int32 => {
            let data = tensor.data().as_i32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Int64 => {
            let data = tensor.data().as_i64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Bool => {
            let data = tensor.data().as_bool_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
    }
}

fn convert_tensor_to_python_scalar(tensor: &Tensor, py: Python) -> PyResult<Py<PyAny>> {
    if tensor.numel() != 1 {
        return Err(PyErr::new::<PyRuntimeError, _>(format!(
            "a Tensor with {} elements cannot be converted to Scalar",
            tensor.numel()
        )));
    }

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor.data().as_f32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Float64 => {
            let data = tensor.data().as_f64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Int32 => {
            let data = tensor.data().as_i32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Int64 => {
            let data = tensor.data().as_i64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Bool => {
            let data = tensor.data().as_bool_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
            })?;
            data[0].into_py_any(py)
        }
    }
}

fn nested_list_from_slice<'py, T>(
    py: Python<'py>,
    data: &[T],
    shape: &[usize],
) -> PyResult<Py<PyAny>>
where
    T: Copy + IntoPyObjectExt<'py>,
{
    if shape.is_empty() {
        if let Some(value) = data.first() {
            return (*value).into_py_any(py);
        }
        return PyList::empty(py).into_py_any(py);
    }

    if shape.len() == 1 {
        let mut elements: Vec<Py<PyAny>> = Vec::with_capacity(data.len());
        for value in data.iter().copied() {
            elements.push(value.into_py_any(py)?);
        }
        let list = PyList::new(py, elements)?;
        return list.into_py_any(py);
    }

    let chunk = shape[1..]
        .iter()
        .fold(1usize, |acc, &dim| acc.saturating_mul(dim));
    let mut parts: Vec<Py<PyAny>> = Vec::with_capacity(shape[0]);
    for index in 0..shape[0] {
        let start = index * chunk;
        let end = start + chunk;
        let slice = if start <= end && end <= data.len() {
            &data[start..end]
        } else {
            &[]
        };
        parts.push(nested_list_from_slice(py, slice, &shape[1..])?);
    }

    let list = PyList::new(py, parts)?;
    list.into_py_any(py)
}

fn create_random_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    normal: bool,
) -> PyResult<Tensor> {
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                use rand::Rng;
                let mut rng = rand::rng();
                if normal {
                    use rand_distr::{Distribution, Normal};
                    let normal_dist = Normal::new(0.0f32, 1.0f32).unwrap();
                    for val in slice.iter_mut() {
                        *val = normal_dist.sample(&mut rng);
                    }
                } else {
                    for val in slice.iter_mut() {
                        *val = rng.random::<f32>();
                    }
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                use rand::Rng;
                let mut rng = rand::rng();
                if normal {
                    use rand_distr::{Distribution, Normal};
                    let normal_dist = Normal::new(0.0f64, 1.0f64).unwrap();
                    for val in slice.iter_mut() {
                        *val = normal_dist.sample(&mut rng);
                    }
                } else {
                    for val in slice.iter_mut() {
                        *val = rng.random::<f64>();
                    }
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                use rand::Rng;
                let mut rng = rand::rng();
                if normal {
                    use rand_distr::{Distribution, Normal};
                    let normal_dist = Normal::new(0.0f32, 1.0f32).unwrap();
                    for val in slice.iter_mut() {
                        *val = normal_dist.sample(&mut rng) as i32;
                    }
                } else {
                    for val in slice.iter_mut() {
                        *val = rng.random::<i32>();
                    }
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                use rand::Rng;
                let mut rng = rand::rng();
                if normal {
                    use rand_distr::{Distribution, Normal};
                    let normal_dist = Normal::new(0.0f64, 1.0f64).unwrap();
                    for val in slice.iter_mut() {
                        *val = normal_dist.sample(&mut rng) as i64;
                    }
                } else {
                    for val in slice.iter_mut() {
                        *val = rng.random::<i64>();
                    }
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                use rand::Rng;
                let mut rng = rand::rng();
                for val in slice.iter_mut() {
                    *val = rng.random::<bool>();
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_eye_tensor(
    n: usize,
    m: usize,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let shape = Shape::new(vec![n, m]);
    let mut tensor_data = TensorData::zeros_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1.0;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1.0;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = true;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_full_tensor(
    shape: Vec<usize>,
    fill_value: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let shape = Shape::new(shape);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                slice.fill(fill_value as f32);
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                slice.fill(fill_value);
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                slice.fill(fill_value as i32);
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                slice.fill(fill_value as i64);
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                slice.fill(fill_value != 0.0);
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_arange_tensor(
    start: f64,
    end: f64,
    step: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if step == 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Step cannot be zero",
        ));
    }

    let num_elements = ((end - start) / step).ceil() as usize;
    let shape = Shape::new(vec![num_elements]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = start + i as f64 * step;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_linspace_tensor(
    start: f64,
    end: f64,
    steps: usize,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of steps must be positive",
        ));
    }

    let shape = Shape::new(vec![steps]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);
    let denom = if steps > 1 { (steps - 1) as f64 } else { 1.0 };
    let step = if steps > 1 {
        (end - start) / denom
    } else {
        0.0
    };

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value.round() as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value.round() as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_logspace_tensor(
    start: f64,
    end: f64,
    steps: usize,
    base: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of steps must be positive",
        ));
    }

    if base <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Base must be positive",
        ));
    }

    let shape = Shape::new(vec![steps]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);
    let denom = if steps > 1 { (steps - 1) as f64 } else { 1.0 };
    let step = if steps > 1 {
        (end - start) / denom
    } else {
        0.0
    };

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent) as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent);
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent).round() as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent).round() as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent) != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}
