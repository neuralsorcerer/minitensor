// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "linalg/diagonal.rs"]
mod diagonal_impl;
#[path = "linalg/matmul.rs"]
mod matmul_impl;
#[path = "linalg/triangular.rs"]
mod triangular_impl;

pub use self::diagonal_impl::*;
pub use self::matmul_impl::*;
pub(crate) use self::triangular_impl::*;
