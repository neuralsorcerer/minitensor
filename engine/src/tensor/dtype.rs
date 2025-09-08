// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported data types for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DataType {
    #[default]
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl DataType {
    /// Get the size in bytes of this data type
    #[inline(always)]
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }

    /// Alias for size_bytes for compatibility
    #[inline(always)]
    pub fn size_in_bytes(&self) -> usize {
        self.size_bytes()
    }

    /// Check if this is a floating point type
    #[inline(always)]
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// Check if this is an integer type
    #[inline(always)]
    pub fn is_int(&self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64)
    }

    /// Check if this is a boolean type
    #[inline(always)]
    pub fn is_bool(&self) -> bool {
        matches!(self, DataType::Bool)
    }

    /// Get the name of this data type as a string
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Bool => "bool",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for types that can be used as tensor elements
pub trait TensorElement: Copy + Clone + Send + Sync + 'static {
    const DTYPE: DataType;
}

impl TensorElement for f32 {
    const DTYPE: DataType = DataType::Float32;
}

impl TensorElement for f64 {
    const DTYPE: DataType = DataType::Float64;
}

impl TensorElement for i32 {
    const DTYPE: DataType = DataType::Int32;
}

impl TensorElement for i64 {
    const DTYPE: DataType = DataType::Int64;
}

impl TensorElement for bool {
    const DTYPE: DataType = DataType::Bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_properties() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float64.size_bytes(), 8);
        assert_eq!(DataType::Int32.size_bytes(), 4);
        assert_eq!(DataType::Int64.size_bytes(), 8);
        assert_eq!(DataType::Bool.size_bytes(), 1);

        assert!(DataType::Float32.is_float());
        assert!(DataType::Float64.is_float());
        assert!(!DataType::Int32.is_float());

        assert!(DataType::Int32.is_int());
        assert!(DataType::Int64.is_int());
        assert!(!DataType::Float32.is_int());

        assert!(DataType::Bool.is_bool());
        assert!(!DataType::Float32.is_bool());
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DataType::Float32.to_string(), "float32");
        assert_eq!(DataType::Float64.to_string(), "float64");
        assert_eq!(DataType::Int32.to_string(), "int32");
        assert_eq!(DataType::Int64.to_string(), "int64");
        assert_eq!(DataType::Bool.to_string(), "bool");
    }

    #[test]
    fn test_tensor_element_trait() {
        assert_eq!(f32::DTYPE, DataType::Float32);
        assert_eq!(f64::DTYPE, DataType::Float64);
        assert_eq!(i32::DTYPE, DataType::Int32);
        assert_eq!(i64::DTYPE, DataType::Int64);
        assert_eq!(bool::DTYPE, DataType::Bool);
    }
}
