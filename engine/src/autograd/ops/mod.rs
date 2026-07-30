// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient functions recorded by tensor operations, one module per operation
//! family, mirroring the forward clusters in [`crate::ops`]:
//!
//! | module          | covers                                                  |
//! |-----------------|---------------------------------------------------------|
//! | `activation`    | relu/gelu/silu/…, sigmoid, tanh, (masked) softmax       |
//! | `arithmetic`    | +,-,*,/,pow, transcendentals, abs/clamp, cum*/sum/prod  |
//! | `core`          | the graph itself, plus the ops it needs to run backward |
//! | `linalg`        | dot, matmul, solve                                      |
//! | `loss`          | MSE, MAE, Huber, cross-entropy, BCE, KL, focal          |
//! | `pooling`       | conv2d, max/avg pool                                    |
//! | `reduction`     | min/max, median, quantile, norm, layer norm             |
//! | `shape`         | reshape, expand, concat, index/gather/scatter, repeat   |
//!
//! Each `Backward` type is declared in the same module as its
//! [`GradientFunction`] impl. Everything public is re-exported up to
//! `crate::autograd`, so callers keep using `crate::autograd::X` and the
//! grouping stays free to change.

mod activation;
mod arithmetic;
mod core;
mod linalg;
mod loss;
mod pooling;
mod reduction;
mod shape;
#[cfg(test)]
mod tests;

pub use self::activation::*;
pub use self::arithmetic::*;
pub use self::core::*;
pub use self::linalg::*;
pub use self::loss::*;
pub use self::pooling::*;
pub use self::reduction::*;
pub use self::shape::*;
