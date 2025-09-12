// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Memory management utilities for minitensor.
//!
//! This module exposes low-level allocation helpers, pooled allocators and a
//! global memory manager. The implementations favour performance and reuse of
//! previously allocated blocks to reduce system allocator pressure.

pub mod allocator;
pub mod manager;
pub mod pool;

pub use allocator::{Allocator, CpuAllocator};
pub use manager::{UnifiedMemoryManager, global_allocate, global_deallocate, init_memory_manager};
pub use pool::{MemoryPool, PoolStats, PooledAllocator};
