// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use engine::ops::bitwise;

/// The body every bitwise method and operator dunder shares.
///
/// The operands are promoted here with the same `BinaryOpKind` the engine op
/// uses, so a float operand is rejected with the promotion table's message
/// before any buffer is allocated.
///
/// This is a free function rather than a macro around `#[pymethods]`: pyo3's
/// generated dispatch loses its `unsafe` context when it expands inside a
/// `macro_rules!` body, which costs an `unsafe_op_in_unsafe_fn` warning per
/// generated dunder.
fn bitwise_binary(
    lhs: &Tensor,
    other: &Bound<PyAny>,
    reverse: bool,
    kind: BinaryOpKind,
    op: fn(&Tensor, &Tensor) -> engine::Result<Tensor>,
) -> PyResult<PyTensor> {
    let (lhs, rhs) = prepare_binary_operands_from_py(lhs, other, reverse, kind)?;
    let result = op(&lhs, &rhs).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

/// The body every logical method shares.
///
/// These deliberately skip the promotion step the bitwise ops go through: both
/// operands are about to be reduced to truth values, so casting a boolean mask
/// up to its partner's float dtype first would allocate a buffer only to throw
/// it away.
fn logical_binary(
    lhs: &Tensor,
    other: &Bound<PyAny>,
    op: fn(&Tensor, &Tensor) -> engine::Result<Tensor>,
) -> PyResult<PyTensor> {
    let rhs = tensor_from_py_value(lhs, other)?;
    let result = op(lhs, &rhs).map_err(_convert_error)?;
    Ok(PyTensor::from_tensor(result))
}

#[pymethods]
impl PyTensor {
    /// Element-wise bitwise AND, and logical AND for booleans.
    pub fn bitwise_and(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            false,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_and,
        )
    }

    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.bitwise_and(other)
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            true,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_and,
        )
    }

    /// Element-wise bitwise OR, and logical OR for booleans.
    pub fn bitwise_or(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            false,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_or,
        )
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.bitwise_or(other)
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            true,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_or,
        )
    }

    /// Element-wise bitwise XOR, and logical XOR for booleans.
    pub fn bitwise_xor(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            false,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_xor,
        )
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.bitwise_xor(other)
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            true,
            BinaryOpKind::Bitwise,
            bitwise::bitwise_xor,
        )
    }

    /// Element-wise left shift. Counts at or past the dtype's width give 0.
    pub fn bitwise_left_shift(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            false,
            BinaryOpKind::Shift,
            bitwise::bitwise_left_shift,
        )
    }

    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.bitwise_left_shift(other)
    }

    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            true,
            BinaryOpKind::Shift,
            bitwise::bitwise_left_shift,
        )
    }

    /// Element-wise arithmetic right shift, preserving sign. Counts at or past the dtype's width give 0 for non-negative values and -1 for negative ones.
    pub fn bitwise_right_shift(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            false,
            BinaryOpKind::Shift,
            bitwise::bitwise_right_shift,
        )
    }

    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.bitwise_right_shift(other)
    }

    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        bitwise_binary(
            &self.inner,
            other,
            true,
            BinaryOpKind::Shift,
            bitwise::bitwise_right_shift,
        )
    }

    fn __invert__(&self) -> PyResult<Self> {
        let result = bitwise::bitwise_not(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise bitwise complement, and logical NOT for booleans.
    pub fn bitwise_not(&self) -> PyResult<Self> {
        self.__invert__()
    }

    /// Element-wise logical AND over truth values, giving a boolean tensor.
    pub fn logical_and(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        logical_binary(&self.inner, other, bitwise::logical_and)
    }

    /// Element-wise logical OR over truth values, giving a boolean tensor.
    pub fn logical_or(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        logical_binary(&self.inner, other, bitwise::logical_or)
    }

    /// Element-wise logical XOR over truth values, giving a boolean tensor.
    pub fn logical_xor(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        logical_binary(&self.inner, other, bitwise::logical_xor)
    }

    /// Element-wise logical NOT over truth values, giving a boolean tensor.
    pub fn logical_not(&self) -> PyResult<Self> {
        let result = bitwise::logical_not(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}
