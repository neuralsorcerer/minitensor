// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    /// Element-wise equality, giving a boolean tensor.
    // Comparison operations
    pub fn eq(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.eq_from_py(other)
    }

    /// Element-wise inequality, giving a boolean tensor.
    pub fn ne(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ne_from_py(other)
    }

    /// Element-wise `<`, giving a boolean tensor.
    pub fn lt(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.lt_from_py(other)
    }

    /// Element-wise `<=`, giving a boolean tensor.
    pub fn le(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.le_from_py(other)
    }

    /// Element-wise `>`, giving a boolean tensor.
    pub fn gt(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.gt_from_py(other)
    }

    /// Element-wise `>=`, giving a boolean tensor.
    pub fn ge(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ge_from_py(other)
    }

    /// Whether two tensors have the same shape and every element equal. NaN is never equal to itself.
    pub fn array_equal(&self, other: &PyTensor) -> PyResult<bool> {
        if self.inner.shape() != other.inner.shape() {
            return Ok(false);
        }
        // An integer against a float goes through the exact comparison: at
        // their common dtype an int32 `20_000_001` and `20_000_000.0` were the
        // same float32, and two different arrays compared equal.
        if mixes_integer_and_float(&self.inner, &other.inner) {
            let equal = self.inner.eq(&other.inner).map_err(_convert_error)?;
            return Ok(equal
                .data()
                .as_bool_slice()
                .is_some_and(|values| values.iter().all(|&v| v)));
        }
        let (lhs, rhs, _) = coerce_binary_operands(&self.inner, &other.inner, BinaryOpKind::Add)
            .map_err(_convert_error)?;
        Ok(lhs.array_equal(&rhs))
    }

    /// Whether every pair of elements is within `rtol`/`atol`.
    #[pyo3(signature = (other, rtol=None, atol=None, equal_nan=false))]
    pub fn allclose(
        &self,
        other: &PyTensor,
        rtol: Option<f64>,
        atol: Option<f64>,
        equal_nan: bool,
    ) -> PyResult<bool> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);
        if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
            return Err(PyValueError::new_err(
                "rtol and atol must be non-negative, finite values",
            ));
        }
        // An integer against a float is measured in float64, which holds every
        // int32 and float32 exactly, rather than at their common dtype.
        if mixes_integer_and_float(&self.inner, &other.inner) {
            let lhs = self
                .inner
                .astype(DataType::Float64)
                .map_err(_convert_error)?;
            let rhs = other
                .inner
                .astype(DataType::Float64)
                .map_err(_convert_error)?;
            return Ok(lhs.allclose_with_equal_nan(&rhs, rtol, atol, equal_nan));
        }
        let (lhs, rhs, _) = coerce_binary_operands(&self.inner, &other.inner, BinaryOpKind::Add)
            .map_err(_convert_error)?;
        Ok(lhs.allclose_with_equal_nan(&rhs, rtol, atol, equal_nan))
    }
}

/// Whether one operand is an integer tensor and the other a float one -- the
/// pair whose common dtype cannot hold both exactly.
fn mixes_integer_and_float(lhs: &Tensor, rhs: &Tensor) -> bool {
    let integer = |t: &Tensor| matches!(t.dtype(), DataType::Int32 | DataType::Int64);
    let float = |t: &Tensor| matches!(t.dtype(), DataType::Float32 | DataType::Float64);
    (integer(lhs) && float(rhs)) || (float(lhs) && integer(rhs))
}

/// The comparisons themselves, shared with the dunder forms in
/// `comparison_dunder.rs`.
///
/// Outside `#[pymethods]` deliberately. `pub(crate)` is the visibility these
/// were always meant to have, but a `pub(crate) fn` inside a `#[pymethods]`
/// block is still exported to Python -- the attribute decides what Python
/// sees, not the Rust modifier -- so each of these stood as a second public
/// method doing exactly what its one-line wrapper above does.
impl PyTensor {
    pub(crate) fn eq_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.eq(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub(crate) fn ne_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.ne(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub(crate) fn lt_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.lt(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub(crate) fn le_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.le(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub(crate) fn gt_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.gt(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub(crate) fn ge_from_py(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) = comparison_operands_from_py(&self.inner, other, false)?;
        let result = lhs.ge(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}
