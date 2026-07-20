// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Automatic differentiation: tape building, backward planning/execution,
//! and the gradient functions recorded by tensor operations.
//!
//! `graph` holds the tape plus the backward planner/executor; `ops` holds the
//! gradient functions grouped by operation family. Everything public is
//! re-exported here so callers keep using `crate::autograd::X`.

pub mod graph;
pub mod ops;

pub use self::ops::*;
