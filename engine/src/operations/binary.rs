// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Tensor},
};
use std::borrow::Cow;

/// Binary operation kinds that influence dtype promotion rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

/// Cast the two operands to a common dtype according to PyTorch style
/// promotion rules for the supplied binary operation.
pub fn coerce_binary_operands<'a>(
    lhs: &'a Tensor,
    rhs: &'a Tensor,
    op: BinaryOpKind,
) -> Result<(Cow<'a, Tensor>, Cow<'a, Tensor>, DataType)> {
    let result_dtype = result_dtype_for_binary_op(lhs.dtype(), rhs.dtype(), op)?;
    let lhs_cast = cast_tensor_to_dtype(lhs, result_dtype)?;
    let rhs_cast = cast_tensor_to_dtype(rhs, result_dtype)?;
    Ok((lhs_cast, rhs_cast, result_dtype))
}

fn cast_tensor_to_dtype<'a>(tensor: &'a Tensor, dtype: DataType) -> Result<Cow<'a, Tensor>> {
    if tensor.dtype() == dtype {
        Ok(Cow::Borrowed(tensor))
    } else {
        Ok(Cow::Owned(tensor.astype(dtype)?))
    }
}

fn result_dtype_for_binary_op(lhs: DataType, rhs: DataType, op: BinaryOpKind) -> Result<DataType> {
    use BinaryOpKind::*;
    match op {
        Add | Mul | Maximum | Minimum => Ok(promote_arithmetic_dtype(lhs, rhs)),
        Sub => {
            if lhs == DataType::Bool || rhs == DataType::Bool {
                Err(MinitensorError::invalid_operation(
                    "Subtraction not supported for boolean tensors",
                ))
            } else {
                Ok(promote_arithmetic_dtype(lhs, rhs))
            }
        }
        Div => Ok(promote_division_dtype(lhs, rhs)),
    }
}

fn promote_arithmetic_dtype(lhs: DataType, rhs: DataType) -> DataType {
    use DataType::*;

    match (lhs, rhs) {
        (Bool, Bool) => Bool,
        (Bool, other) => other,
        (other, Bool) => other,
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) => Float32,
        (Int64, _) | (_, Int64) => Int64,
        (Int32, Int32) => Int32,
    }
}

fn promote_division_dtype(lhs: DataType, rhs: DataType) -> DataType {
    if lhs == DataType::Float64 || rhs == DataType::Float64 {
        DataType::Float64
    } else {
        DataType::Float32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_arithmetic_types() {
        assert_eq!(
            promote_arithmetic_dtype(DataType::Int32, DataType::Float32),
            DataType::Float32
        );
        assert_eq!(
            promote_arithmetic_dtype(DataType::Int32, DataType::Int64),
            DataType::Int64
        );
        assert_eq!(
            promote_arithmetic_dtype(DataType::Bool, DataType::Int32),
            DataType::Int32
        );
        assert_eq!(
            promote_arithmetic_dtype(DataType::Bool, DataType::Bool),
            DataType::Bool
        );
    }

    #[test]
    fn test_promote_division_types() {
        assert_eq!(
            promote_division_dtype(DataType::Int32, DataType::Int32),
            DataType::Float32
        );
        assert_eq!(
            promote_division_dtype(DataType::Float64, DataType::Int64),
            DataType::Float64
        );
    }
}
