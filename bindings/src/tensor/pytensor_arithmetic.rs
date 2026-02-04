// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
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

}
