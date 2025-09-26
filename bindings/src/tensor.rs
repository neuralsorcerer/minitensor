// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::PyDevice;
use crate::error::_convert_error;
use engine::operations::binary::{BinaryOpKind, coerce_binary_operands};
use engine::operations::shape_ops::RepeatInterleaveSpec;
use engine::tensor::{Shape, TensorData};
use engine::{DataType, Device, MinitensorError, Tensor, TensorIndex};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PySlice, PyTuple};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::sync::Arc;

/// Python wrapper for Tensor with comprehensive functionality
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
        Self { inner: tensor }
    }
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from Python data
    #[new]
    fn new(
        data: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = convert_python_data_to_tensor(data, dtype, device, requires_grad)?;
        Ok(Self { inner: tensor })
    }

    // Properties
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype()).to_lowercase()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn grad(&self) -> Option<Self> {
        self.inner.grad().map(|g| Self {
            inner: (**g).clone(),
        })
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
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().as_slice().to_vec()
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
    fn reshape(&self, shape: Vec<isize>) -> PyResult<Self> {
        let reshaped = engine::operations::reshape_with_inference(&self.inner, shape)
            .map_err(_convert_error)?;
        Ok(Self { inner: reshaped })
    }

    fn transpose(&self, dim0: Option<isize>, dim1: Option<isize>) -> PyResult<Self> {
        let dim0 = dim0.unwrap_or(0);
        let dim1 = dim1.unwrap_or(1);
        let result = self.inner.transpose(dim0, dim1).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn permute(&self, dims: Vec<isize>) -> PyResult<Self> {
        let result = self.inner.permute(dims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn movedim(&self, source: &Bound<PyAny>, destination: &Bound<PyAny>) -> PyResult<Self> {
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
        Ok(Self { inner: result })
    }

    fn squeeze(&self, dim: Option<isize>) -> PyResult<Self> {
        let result = if let Some(d) = dim {
            self.inner.squeeze_dim(d)
        } else {
            self.inner.squeeze()
        }
        .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.unsqueeze(dim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn expand(&self, dims: Vec<isize>) -> PyResult<Self> {
        let result = self.inner.expand(dims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn repeat(&self, repeats: Vec<usize>) -> PyResult<Self> {
        let result = self.inner.repeat(repeats).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn flip(&self, dims: Vec<isize>) -> PyResult<Self> {
        let result =
            engine::operations::shape_ops::flip(&self.inner, &dims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn roll(&self, shifts: Vec<isize>, dims: Option<Vec<isize>>) -> PyResult<Self> {
        let dims_ref = dims.as_ref().map(|d| d.as_slice());
        let result = engine::operations::shape_ops::roll(&self.inner, &shifts, dims_ref)
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn repeat_interleave(
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
            return Ok(Self { inner: result });
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
            return Ok(Self { inner: result });
        }

        if let Ok(py_tensor) = repeats.extract::<PyRef<PyTensor>>() {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Tensor(py_tensor.tensor()),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self { inner: result });
        }

        Err(PyTypeError::new_err(
            "repeat_interleave: repeats must be an int, sequence of ints, or Tensor",
        ))
    }

    fn narrow(&self, dim: isize, start: usize, length: usize) -> PyResult<Self> {
        let result = engine::operations::shape_ops::narrow(&self.inner, dim, start, length)
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn flatten(&self, start_dim: Option<isize>, end_dim: Option<isize>) -> PyResult<Self> {
        let ndim = self.ndim() as isize;
        let start = start_dim.unwrap_or(0);
        let end = end_dim.unwrap_or(ndim - 1);

        let result = self.inner.flatten(start, end).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn ravel(&self) -> PyResult<Self> {
        self.flatten(None, None)
    }

    // Tensor operations
    fn clone(&self) -> PyResult<Self> {
        let result = self.inner.deep_clone().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    fn contiguous(&self) -> PyResult<Self> {
        let result = self.inner.contiguous().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn to(&self, device: &PyDevice) -> PyResult<Self> {
        let result = self.inner.to(device.device()).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn cpu(&self) -> PyResult<Self> {
        let result = self.inner.to(Device::cpu()).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn astype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = parse_dtype(dtype)?;
        let result = self.inner.astype(dtype).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    // Gradient operations
    fn backward(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        let grad = gradient.map(|g| g.inner.clone());
        self.inner.backward(grad).map_err(_convert_error)?;
        Ok(())
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        self.inner = self.inner.clone().requires_grad_(requires_grad);
        Ok(())
    }

    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        self.inner.zero_grad(set_to_none.unwrap_or(false));
    }

    // Arithmetic operations
    fn __neg__(&self) -> PyResult<Self> {
        use engine::operations::arithmetic::neg;
        let result = neg(&self.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.add(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        use engine::operations::arithmetic::sub;
        let result = sub(&self.inner, &other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn __mul__(&self, other: &PyTensor) -> PyResult<Self> {
        use engine::operations::arithmetic::mul;
        let result = mul(&self.inner, &other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<Self> {
        use engine::operations::arithmetic::div;
        let result = div(&self.inner, &other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    // Comparison operators as Python dunder methods
    fn __eq__(&self, other: &PyTensor) -> PyResult<Self> {
        self.eq(other)
    }

    fn __ne__(&self, other: &PyTensor) -> PyResult<Self> {
        self.ne(other)
    }

    fn __lt__(&self, other: &PyTensor) -> PyResult<Self> {
        self.lt(other)
    }

    fn __le__(&self, other: &PyTensor) -> PyResult<Self> {
        self.le(other)
    }

    fn __gt__(&self, other: &PyTensor) -> PyResult<Self> {
        self.gt(other)
    }

    fn __ge__(&self, other: &PyTensor) -> PyResult<Self> {
        self.ge(other)
    }

    pub fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.matmul(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn triu(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.triu(diagonal).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn tril(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.tril(diagonal).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    #[pyo3(name = "where")]
    pub fn where_method(&self, condition: &PyTensor, other: &PyTensor) -> PyResult<Self> {
        let result = self
            .inner
            .where_select(&condition.inner, &other.inner)
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn masked_fill(&self, mask: &PyTensor, value: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(tensor_value) = value.extract::<PyTensor>() {
            let result = self
                .inner
                .masked_fill(&mask.inner, &tensor_value.inner)
                .map_err(_convert_error)?;
            Ok(Self { inner: result })
        } else {
            let scalar = if let Ok(v) = value.extract::<f64>() {
                v
            } else if let Ok(v) = value.extract::<i64>() {
                v as f64
            } else if let Ok(v) = value.extract::<bool>() {
                if v { 1.0 } else { 0.0 }
            } else {
                return Err(PyTypeError::new_err(
                    "masked_fill value must be a Tensor or numeric scalar",
                ));
            };

            let result = self
                .inner
                .masked_fill_scalar(&mask.inner, scalar)
                .map_err(_convert_error)?;
            Ok(Self { inner: result })
        }
    }

    pub fn maximum(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.maximum(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn minimum(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.minimum(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn logaddexp(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.logaddexp(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
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
    pub fn eq(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.eq(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn ne(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.ne(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn lt(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.lt(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn le(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.le(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn gt(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.gt(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn ge(&self, other: &PyTensor) -> PyResult<Self> {
        let result = self.inner.ge(&other.inner).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    // Reduction operations
    pub fn sum(&self, dim: Option<Vec<isize>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.sum(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn logsumexp(&self, dim: Option<Vec<isize>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.logsumexp(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn prod(&self, dim: Option<Vec<isize>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.prod(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn mean(&self, dim: Option<Vec<isize>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.mean(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn all(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.all(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn any(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.any(dim, keepdim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn cumsum(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumsum(dim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn cumprod(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumprod(dim).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn max(&self, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let result = self.inner.max(axis, keepdims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn min(&self, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let result = self.inner.min(axis, keepdims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn median(
        &self,
        axis: Option<isize>,
        keepdims: Option<bool>,
    ) -> PyResult<(Self, Option<Self>)> {
        let keepdims = keepdims.unwrap_or(false);
        match self.inner.median(axis, keepdims) {
            Ok((values, indices_opt)) => {
                let values_tensor = Self { inner: values };
                let indices_tensor = indices_opt.map(|inner| Self { inner });
                Ok((values_tensor, indices_tensor))
            }
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    pub fn argmax(&self, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let result = self.inner.argmax(axis, keepdims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn argmin(&self, axis: Option<isize>, keepdims: Option<bool>) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let result = self.inner.argmin(axis, keepdims).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

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
            Ok((values, indices)) => Ok((Self { inner: values }, Self { inner: indices })),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    pub fn sort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<(Self, Self)> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.sort(dim, descending, stable) {
            Ok((values, indices)) => Ok((Self { inner: values }, Self { inner: indices })),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    pub fn argsort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<Self> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.argsort(dim, descending, stable) {
            Ok(indices) => Ok(Self { inner: indices }),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    pub fn std(
        &self,
        axis: Option<isize>,
        keepdims: Option<bool>,
        unbiased: Option<bool>,
    ) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .std(axis, keepdims, unbiased)
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    pub fn var(
        &self,
        axis: Option<isize>,
        keepdims: Option<bool>,
        unbiased: Option<bool>,
    ) -> PyResult<Self> {
        let keepdims = keepdims.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .var(axis, keepdims, unbiased)
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    // Mathematical functions
    fn abs(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn sqrt(&self) -> PyResult<Self> {
        let result = self.inner.sqrt().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn rsqrt(&self) -> PyResult<Self> {
        let result = self.inner.rsqrt().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn pow(&self, exponent: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(exp_tensor) = exponent.extract::<PyTensor>() {
            let result = self.inner.pow(&exp_tensor.inner).map_err(_convert_error)?;
            Ok(Self { inner: result })
        } else {
            let exp = exponent.extract::<f64>()?;
            let result = self.inner.powf(exp).map_err(_convert_error)?;
            Ok(Self { inner: result })
        }
    }

    fn exp(&self) -> PyResult<Self> {
        let result = self.inner.exp().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn log(&self) -> PyResult<Self> {
        let result = self.inner.log().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn log1p(&self) -> PyResult<Self> {
        let result = self.inner.log1p().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn expm1(&self) -> PyResult<Self> {
        let result = self.inner.expm1().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn sin(&self) -> PyResult<Self> {
        let result = self.inner.sin().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn cos(&self) -> PyResult<Self> {
        let result = self.inner.cos().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn tan(&self) -> PyResult<Self> {
        let result = self.inner.tan().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn clamp(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        let result = self.inner.clamp(min, max).map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn isnan(&self) -> PyResult<Self> {
        let result = self.inner.isnan().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn isinf(&self) -> PyResult<Self> {
        let result = self.inner.isinf().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn isfinite(&self) -> PyResult<Self> {
        let result = self.inner.isfinite().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn __pow__(&self, exponent: &Bound<PyAny>, _mod: Option<&Bound<PyAny>>) -> PyResult<Self> {
        self.pow(exponent)
    }

    fn relu(&self) -> PyResult<Self> {
        let result = self.inner.relu().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn hardshrink(&self, lambd: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .hardshrink(lambd.unwrap_or(0.5))
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn softmax(&self, dim: Option<isize>) -> PyResult<Self> {
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
        Ok(Self { inner: result })
    }

    fn log_softmax(&self, dim: Option<isize>) -> PyResult<Self> {
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
        Ok(Self { inner: result })
    }

    fn layer_norm(
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
        Ok(Self { inner: result })
    }

    fn gelu(&self, approximate: Option<&str>) -> PyResult<Self> {
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
        Ok(Self { inner: result })
    }

    fn sigmoid(&self) -> PyResult<Self> {
        let result = self.inner.sigmoid().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn softplus(&self, beta: Option<f64>, threshold: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .softplus(beta.unwrap_or(1.0), threshold.unwrap_or(20.0))
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn elu(&self, alpha: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .elu(alpha.unwrap_or(1.0))
            .map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn selu(&self) -> PyResult<Self> {
        let result = self.inner.selu().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn silu(&self) -> PyResult<Self> {
        let result = self.inner.silu().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn softsign(&self) -> PyResult<Self> {
        let result = self.inner.softsign().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    fn tanh(&self) -> PyResult<Self> {
        let result = self.inner.tanh().map_err(_convert_error)?;
        Ok(Self { inner: result })
    }

    // NumPy conversion methods
    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, false)
    }

    fn numpy_copy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, true)
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
        Ok(Self { inner: result })
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
    pub fn empty(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(shape);
        let tensor = Tensor::empty(shape, dtype, device, requires_grad);
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    pub fn zeros(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(shape);
        let tensor = Tensor::zeros(shape, dtype, device, requires_grad);
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    pub fn ones(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(shape);
        let tensor = Tensor::ones(shape, dtype, device, requires_grad);
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn rand(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(shape);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, false)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn randn(
        shape: Vec<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(shape);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, true)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let m = m.unwrap_or(n);
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_eye_tensor(n, m, dtype, device, requires_grad)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    pub fn full(
        shape: Vec<usize>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_full_tensor(shape, fill_value, dtype, device, requires_grad)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn arange(
        start: f64,
        end: f64,
        step: Option<f64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let step = step.unwrap_or(1.0);
        let dtype = parse_dtype(dtype.unwrap_or("float32"))?;
        let device = device.map(|d| d.device()).unwrap_or_else(|| Device::cpu());
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_arange_tensor(start, end, step, dtype, device, requires_grad)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn from_numpy(array: &Bound<PyAny>, requires_grad: Option<bool>) -> PyResult<Self> {
        let requires_grad = requires_grad.unwrap_or(false);
        let tensor = convert_numpy_to_tensor(array, requires_grad)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    fn from_numpy_shared(array: &Bound<PyAny>, requires_grad: Option<bool>) -> PyResult<Self> {
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
            .map(|obj| obj.extract::<PyTensor>().map(|t| t.inner.clone()))
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
                let t = obj.extract::<PyTensor>()?;
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
    pub fn chunk(&self, sections: usize, axis: Option<isize>) -> PyResult<Vec<PyTensor>> {
        if sections <= 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sections must be greater than zero",
            ));
        }

        let axis = axis.unwrap_or(0);
        let ndim = self.inner.ndim() as isize;
        let axis = if axis < 0 { axis + ndim } else { axis };
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
    pub fn split(
        &self,
        split_size_or_sections: &Bound<PyAny>,
        axis: Option<isize>,
    ) -> PyResult<Vec<PyTensor>> {
        let axis = axis.unwrap_or(0);
        let ndim = self.inner.ndim() as isize;
        let axis = if axis < 0 { axis + ndim } else { axis };
        if axis < 0 || axis >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} out of range",
                axis
            )));
        }
        let axis = axis as usize;
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

// Helper functions
fn parse_dtype(dtype_str: &str) -> PyResult<DataType> {
    match dtype_str.to_lowercase().as_str() {
        "float32" | "f32" => Ok(DataType::Float32),
        "float64" | "f64" | "double" => Ok(DataType::Float64),
        "int32" | "i32" => Ok(DataType::Int32),
        "int64" | "i64" | "long" => Ok(DataType::Int64),
        "bool" | "boolean" => Ok(DataType::Bool),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported dtype: {}",
            dtype_str
        ))),
    }
}

fn convert_python_data_to_tensor(
    data: &Bound<PyAny>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    // First try NumPy array conversion for any supported dtype
    if let Ok(tensor) = convert_numpy_to_tensor(data, requires_grad) {
        let tensor = if tensor.dtype() != dtype {
            tensor.astype(dtype).map_err(_convert_error)?
        } else {
            tensor
        };
        return Ok(tensor);
    }

    // Handle Python lists and tuples by flattening to float32 then casting
    if let Ok(list) = data.downcast::<PyList>() {
        let (shape, flat_data) = flatten_python_data(list)?;
        let base_data = Arc::new(TensorData::from_vec(flat_data, DataType::Float32, device));
        let mut tensor = Tensor::new(
            base_data,
            Shape::new(shape),
            DataType::Float32,
            device,
            requires_grad,
        );
        if dtype != DataType::Float32 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    // Handle scalars
    if let Ok(val) = data.extract::<f64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec(
            vec![val as f32],
            DataType::Float32,
            device,
        ));
        let mut tensor = Tensor::new(base_data, shape, DataType::Float32, device, requires_grad);
        if dtype != DataType::Float32 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    Err(PyErr::new::<PyTypeError, _>(
        "Unsupported data type for tensor creation",
    ))
}

fn flatten_python_data(list: &Bound<PyList>) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let mut shape = vec![];
    let mut flat_data = vec![];

    fn process_nested(
        item: &Bound<PyAny>,
        shape: &mut Vec<usize>,
        flat_data: &mut Vec<f32>,
        depth: usize,
    ) -> PyResult<()> {
        if let Ok(nested_list) = item.downcast::<PyList>() {
            if depth >= shape.len() {
                shape.push(nested_list.len());
            }
            for nested_item in nested_list.iter() {
                process_nested(&nested_item, shape, flat_data, depth + 1)?;
            }
        } else if let Ok(val) = item.extract::<f64>() {
            flat_data.push(val as f32);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Invalid data type in nested list",
            ));
        }
        Ok(())
    }

    shape.push(list.len());
    for item in list.iter() {
        process_nested(&item, &mut shape, &mut flat_data, 1)?;
    }

    Ok((shape, flat_data))
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
