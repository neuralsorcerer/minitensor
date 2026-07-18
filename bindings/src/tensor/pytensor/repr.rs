// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
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

    fn __abs__(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __pos__(&self) -> Self {
        // `+t` is the identity; the clone shares storage via Arc, matching
        // torch's behavior of returning the input values unchanged.
        self.clone()
    }

    fn __float__(&self) -> PyResult<f64> {
        let value = self.scalar_as_f64()?;
        Ok(value)
    }

    fn __int__(&self) -> PyResult<i64> {
        let value = self.scalar_as_f64()?;
        Ok(value as i64)
    }

    fn __getitem__(&self, key: &Bound<PyAny>) -> PyResult<Self> {
        // NumPy-style fancy forms first: boolean masks select blocks along
        // the leading dims, 1-D integer keys select rows along dim 0.
        if let Some(result) = try_fancy_index_tensor(&self.inner, key)? {
            return Ok(Self::from_tensor(result));
        }
        let (indices, newaxis_positions) = parse_getitem_indices(key, self.inner.shape().dims())?;
        let mut result = self.inner.index(&indices).map_err(_convert_error)?;
        for &pos in &newaxis_positions {
            result = result.unsqueeze(pos as isize).map_err(_convert_error)?;
        }
        Ok(Self::from_tensor(result))
    }

    fn __setitem__(&mut self, key: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<()> {
        // Boolean-mask assignment (`t[mask] = scalar`): the mask must match
        // the leading dimensions; it is padded with trailing length-1 axes so
        // masked_fill's broadcasting applies it to whole blocks.
        if let Some(mask) = try_bool_mask_key(key)? {
            let in_dims = self.inner.shape().dims();
            let m_dims = mask.shape().dims().to_vec();
            if m_dims.len() > in_dims.len() || in_dims[..m_dims.len()] != *m_dims {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "boolean index mask shape {:?} must match the leading dimensions of tensor shape {:?}",
                    m_dims, in_dims
                )));
            }
            let scalar = if let Ok(b) = value.extract::<bool>() {
                b as u8 as f64
            } else if let Ok(v) = value.extract::<f64>() {
                v
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "boolean-mask assignment currently supports scalar values only",
                ));
            };
            let mut padded = m_dims;
            padded.resize(in_dims.len(), 1);
            let mask_padded =
                engine::operations::shape_ops::reshape(&mask, engine::tensor::Shape::new(padded))
                    .map_err(_convert_error)?;
            let filled = engine::operations::selection::masked_fill_scalar(
                &self.inner,
                &mask_padded,
                scalar,
            )
            .map_err(_convert_error)?;
            self.inner.copy_(&filled).map_err(_convert_error)?;
            return Ok(());
        }
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
}

impl PyTensor {
    /// Extract the value of a one-element tensor as f64 for `__float__` /
    /// `__int__`. Mirrors `__bool__`'s single-element requirement.
    fn scalar_as_f64(&self) -> PyResult<f64> {
        if self.inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only one element tensors can be converted to Python scalars",
            ));
        }
        let err =
            || PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to access tensor data");
        match self.inner.dtype() {
            DataType::Float32 => Ok(self.inner.data().as_f32_slice().ok_or_else(err)?[0] as f64),
            DataType::Float64 => Ok(self.inner.data().as_f64_slice().ok_or_else(err)?[0]),
            DataType::Int32 => Ok(self.inner.data().as_i32_slice().ok_or_else(err)?[0] as f64),
            DataType::Int64 => Ok(self.inner.data().as_i64_slice().ok_or_else(err)?[0] as f64),
            DataType::Bool => Ok(if self.inner.data().as_bool_slice().ok_or_else(err)?[0] {
                1.0
            } else {
                0.0
            }),
        }
    }
}
