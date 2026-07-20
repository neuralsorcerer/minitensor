// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient functions recorded by tensor operations, grouped by operation
//! family. These are real submodules (one file per `Backward` family);
//! everything public is re-exported up to `crate::autograd` so callers keep
//! using `crate::autograd::X`.

mod activation;
mod arithmetic;
mod core;
mod linalg;
mod reduction;
mod shape;
#[cfg(test)]
mod tests;

pub use self::activation::*;
pub use self::arithmetic::*;
pub use self::core::*;
pub use self::linalg::*;
pub use self::reduction::*;
pub use self::shape::*;
