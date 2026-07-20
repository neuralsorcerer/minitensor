// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "shape_ops/indexing.rs"]
mod indexing_impl;
#[path = "shape_ops/reshape.rs"]
mod reshape_impl;

pub use self::indexing_impl::*;
pub use self::reshape_impl::*;
