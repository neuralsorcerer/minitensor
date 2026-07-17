// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "loss/classification.rs"]
mod classification_impl;
#[path = "loss/regression.rs"]
mod regression_impl;

pub(crate) use self::classification_impl::*;
pub use self::regression_impl::*;
