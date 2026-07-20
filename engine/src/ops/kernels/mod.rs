// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Dtype-generic kernel bodies, macro-generated.
//!
//! Each submodule holds the per-dtype implementations shared by the
//! higher-level `ops` families. Keeping the generated bodies here (rather than
//! next to the op that first needed them) gives every family one place to reach
//! for a broadcasting/elementwise kernel.

mod binary;

pub(crate) use self::binary::*;
