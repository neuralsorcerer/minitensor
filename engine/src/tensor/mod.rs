// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Tensor type, storage, shapes, and dtype definitions.
//!
//! `core` declares the `Tensor` struct and hosts the method impls (split
//! across its child modules so they retain access to the private fields);
//! everything public is re-exported here so callers keep using
//! `crate::tensor::X`.

pub mod data;
pub mod dtype;
pub mod shape;

#[path = "mod/core.rs"]
mod core;

pub use self::core::*;
