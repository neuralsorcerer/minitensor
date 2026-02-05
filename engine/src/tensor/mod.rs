// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod data;
pub mod dtype;
pub mod shape;
include!("mod/core.rs");
include!("mod/autograd.rs");
include!("mod/ops.rs");
include!("mod/indexing.rs");
include!("mod/utils.rs");
