// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Element-wise arithmetic ops. The dtype-specialized kernel bodies these
//! delegate to live in [`crate::ops::kernels`].

#[path = "arithmetic/elementwise.rs"]
mod elementwise_impl;

pub use self::elementwise_impl::*;
