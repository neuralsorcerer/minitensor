// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Tensor type, storage, shapes, and dtype definitions.
//!
//! `tensor` declares the `Tensor` struct and hosts every method impl in one
//! module (so they retain access to its private fields); `storage` holds the
//! `TensorData` backing buffer. Everything public is re-exported here so
//! callers keep using `crate::tensor::X`.

pub mod dtype;
pub mod shape;
pub mod storage;
#[allow(clippy::module_inception)]
pub mod tensor;

pub use self::tensor::*;
