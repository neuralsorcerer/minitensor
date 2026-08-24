// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod activation;
pub mod arithmetic;
pub mod attention;
pub mod binary;
pub mod comparison;
pub mod conv;
pub mod einsum;
pub mod interpolate;
pub(crate) mod kernels;
pub mod linalg;
pub mod loss;
pub(crate) mod map;
pub mod minmax;
pub mod normalization;
pub mod pooling;
pub mod reduction;
pub mod search;
pub mod selection;
pub mod shape_ops;
pub mod simd;
pub mod unique;
pub(crate) mod util;

// Re-export common operations
pub use activation::*;
pub use arithmetic::*;
pub use attention::*;
pub use comparison::*;
pub use conv::*;
pub use einsum::*;
pub use interpolate::*;
pub use linalg::*;
pub use loss::*;
pub use minmax::*;
pub use normalization::*;
pub use pooling::*;
pub use reduction::*;
pub use search::*;
pub use selection::*;
pub use shape_ops::*;
pub use simd::*;
pub use unique::*;

// Exported so the Python bindings resolve `dim` arguments the same way the
// engine does, and report a bad one with the same message.
pub use util::{normalize_dim, normalize_dim_named};
