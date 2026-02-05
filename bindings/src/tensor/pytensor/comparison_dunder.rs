// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
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

    pub fn solve(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        let rhs_tensor = tensor_from_py_value(&self.inner, rhs)?;
        let result = self.inner.solve(&rhs_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn bmm(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.bmm(&other_tensor).map_err(_convert_error)?;
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

}
