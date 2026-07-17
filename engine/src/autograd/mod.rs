// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Automatic differentiation: tape building, backward planning/execution,
//! and the gradient functions recorded by tensor operations.
//!
//! The submodules group gradient functions by operation family; everything
//! public is re-exported here so callers keep using `crate::autograd::X`.

pub mod graph;

#[path = "mod/activation.rs"]
mod activation;
#[path = "mod/arithmetic.rs"]
mod arithmetic;
#[path = "mod/core.rs"]
mod core;
#[path = "mod/linalg.rs"]
mod linalg;
#[path = "mod/reduction.rs"]
mod reduction;
#[path = "mod/shape.rs"]
mod shape;
#[cfg(test)]
#[path = "mod/tests.rs"]
mod tests;

pub use self::activation::*;
pub use self::arithmetic::*;
pub use self::core::*;
pub use self::linalg::*;
pub use self::reduction::*;
pub use self::shape::*;
