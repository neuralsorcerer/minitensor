// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Python tensor bindings.
//!
//! `preamble` declares `PyTensor` and the shared conversion helpers; the
//! method impls and creation/interop functions live in its child modules so
//! they keep access to the private `inner` field and the shared imports.

#[path = "tensor/preamble.rs"]
mod preamble;

pub use self::preamble::*;
