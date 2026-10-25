// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    // Arithmetic operations
    fn __neg__(&self) -> PyResult<Self> {
        use engine::ops::arithmetic::neg;
        let result = neg(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = lhs.add(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = lhs.add(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::sub;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = sub(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::sub;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = sub(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::mul;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = mul(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::mul;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = mul(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __truediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::div;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rtruediv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::div;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::floor_div;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = floor_div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::floor_div;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = floor_div(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::remainder;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = remainder(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::remainder;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, true)?;
        let result = remainder(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise division rounded towards negative infinity, matching Python's `//`.
    pub fn floor_divide(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__floordiv__(other)
    }

    /// Element-wise modulo taking the sign of the divisor, matching Python's `%`.
    pub fn remainder(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mod__(other)
    }

    /// Element-wise modulo taking the sign of the dividend, matching C's `fmod`.
    pub fn fmod(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        use engine::ops::arithmetic::fmod;
        let (lhs, rhs) = arithmetic_operands_from_py(&self.inner, other, false)?;
        let result = fmod(&lhs, &rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}
