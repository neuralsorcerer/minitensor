// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
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
        if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
            return Err(PyValueError::new_err(
                "rtol and atol must be non-negative, finite values",
            ));
        }
        Ok(self.inner.allclose(&other.inner, rtol, atol))
    }

}
