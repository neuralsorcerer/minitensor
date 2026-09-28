// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Deterministic global random number generation utilities.
//!
//! The Rust standard `rand::rng()` helper returns a thread-local generator that
//! cannot be deterministically seeded.  MiniTensor needs reproducible randomness
//! across the entire stack so that Python callers can rely on `manual_seed`.
//! This module owns a single
//! [`Xoshiro256PlusPlus`] protected by a `Mutex` and exposes helpers that allow
//! the rest of the engine to draw random numbers while sharing the global
//! state.
//!
//! Xoshiro256++ rather than `StdRng`, which is ChaCha12: a cryptographic
//! generator, and at about 2.6ns a word on this crate's baseline x86-64 target
//! it was what `rand`, `randn`, dropout and every initializer spent most of
//! their time in. Nothing here needs a cryptographic stream. Named directly
//! rather than as `SmallRng`, which is a different generator on 32-bit
//! targets, so a seed gives the same numbers on every platform.

use parking_lot::Mutex;
use rand::{RngExt, SeedableRng, rngs::Xoshiro256PlusPlus};
use std::sync::LazyLock;

/// Global RNG used across the engine for deterministic sampling.
static GLOBAL_RNG: LazyLock<Mutex<Xoshiro256PlusPlus>> = LazyLock::new(|| {
    let mut thread_rng = rand::rng();
    let seed = thread_rng.random::<u64>();
    Mutex::new(Xoshiro256PlusPlus::seed_from_u64(seed))
});

/// Execute a closure with exclusive access to the global RNG.
///
/// This helper ensures that all randomness goes through a single generator so
/// callers observe deterministic behaviour once a manual seed is set.  The
/// closure should avoid long-running work while holding the RNG lock.
#[inline]
pub fn with_rng<T>(f: impl FnOnce(&mut Xoshiro256PlusPlus) -> T) -> T {
    let mut guard = GLOBAL_RNG.lock();
    f(&mut guard)
}

/// Seed the global RNG with the provided value.
///
/// Re-seeding resets the generator state so subsequent calls to random tensor
/// creation, dropout masks, and parameter initialisation produce reproducible
/// results.
#[inline]
pub fn manual_seed(seed: u64) {
    *GLOBAL_RNG.lock() = Xoshiro256PlusPlus::seed_from_u64(seed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn test_manual_seed_reproducible() {
        manual_seed(1234);
        let first: Vec<u64> = with_rng(|rng| (0..4).map(|_| rng.random::<u64>()).collect());
        manual_seed(1234);
        let second: Vec<u64> = with_rng(|rng| (0..4).map(|_| rng.random::<u64>()).collect());
        assert_eq!(first, second);
    }
}
