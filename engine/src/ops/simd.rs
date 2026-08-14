// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "simd/kernels.rs"]
mod kernels_impl;
#[path = "simd/transcendental.rs"]
mod transcendental_impl;

pub use self::kernels_impl::*;
pub(crate) use self::transcendental_impl::*;
