// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! A seam for a faster dense GEMM than the engine carries.
//!
//! The engine's own GEMM is a blocked, packed, rayon-parallel kernel, and on
//! everything but one shape of problem it is the right thing to run. The
//! exception is a single large dense product in `float32` or `float64`, where
//! a tuned BLAS wins by 1.6-2.3x on this machine -- decades of per-architecture
//! kernel selection that a portable Rust kernel is not going to match.
//!
//! The library already ships next to one: `numpy`, which every install pulls
//! in, carries a full OpenBLAS. Reaching it costs nothing but a pair of array
//! headers pointed at buffers the engine already owns. That is what a provider
//! is -- the Python bindings install one at import, and a build with no Python
//! in it (`cargo test`, an embedding Rust program) simply has none and runs the
//! engine's kernel, which is why this is a hook and not a dependency.
//!
//! The engine offers the *whole* problem and the provider decides. It sees the
//! dimensions before it commits, so a product too small to be worth a call
//! declines and costs one predictable branch.

use std::sync::OnceLock;

/// How an operand is laid out, relative to the shape the product names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    /// Row-major in the shape the product names it.
    RowMajor,
    /// The same values with the two axes swapped: an operand the product calls
    /// `[r, c]` that is stored `[c, r]`.
    ///
    /// A weight matrix is the reason this exists. `nn.Linear` holds it
    /// `[out, in]` and the product wants `[in, out]`, and a GEMM addresses its
    /// operands by row and column stride, so it reads the matrix where it lies
    /// instead of copying the whole of it on every forward. Anything asked to
    /// stand in for that GEMM has to be able to do the same.
    Transposed,
}

/// One dense product: `out = lhs @ rhs`.
///
/// `lhs` is `[m, k]`, `rhs` is `[k, n]` and `out` is `[m, n]`, each packed with
/// no padding -- the engine materialises every view, so nothing here is strided
/// beyond the two axes being in one order or the other. `out` is row-major,
/// arrives zeroed, and is written rather than accumulated into.
pub struct Gemm<'a, T> {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lhs: &'a [T],
    pub lhs_storage: Storage,
    pub rhs: &'a [T],
    pub rhs_storage: Storage,
    pub out: &'a mut [T],
}

impl<T> Gemm<'_, T> {
    /// Multiply-accumulate count, the size a provider should judge by.
    ///
    /// `m * n * k` rather than the dimensions separately: `1x1024 @ 1024x1`
    /// and `1024x1 @ 1x1024` are the same three numbers and nothing alike as
    /// work, and only the product tells them apart.
    pub fn flops(&self) -> usize {
        self.m.saturating_mul(self.n).saturating_mul(self.k)
    }
}

/// Somewhere better to send a dense GEMM than the engine's own kernel.
///
/// Each method answers whether it ran the product. `false` means the engine
/// runs its own, so declining is always safe and always correct; a provider
/// that cannot help with a particular size or dtype says so rather than
/// arranging to be asked less often.
///
/// A provider must leave `out` exactly as it found it -- zeroed -- when it
/// declines or fails part way, since the fallback writes assuming that.
pub trait GemmProvider: Send + Sync {
    fn gemm_f32(&self, request: Gemm<'_, f32>) -> bool;
    fn gemm_f64(&self, request: Gemm<'_, f64>) -> bool;
}

static PROVIDER: OnceLock<Box<dyn GemmProvider>> = OnceLock::new();

/// Install the provider for the life of the process.
///
/// Returns whether this call was the one that installed it. Once, because a
/// provider is a property of the host rather than of a call -- the bindings
/// install theirs while the extension module is initialising, before any
/// tensor exists -- and because a swap would have to be synchronised against
/// every in-flight GEMM to be sound, at the cost of a lock on the path this
/// exists to make faster.
pub fn set_gemm_provider(provider: Box<dyn GemmProvider>) -> bool {
    PROVIDER.set(provider).is_ok()
}

/// The installed provider, if this build has one and the caller may use it.
///
/// `None` inside a rayon worker, whatever is installed. A provider is free to
/// need something the calling thread does not hold -- the Python one needs the
/// interpreter lock -- and a worker that blocks for it stalls the thread that
/// holds it and is waiting on the join. The engine only ever reaches for a
/// provider on a single whole product, which runs on the caller's own thread;
/// the batched paths that do run under `rayon` are the ones already saturating
/// the pool, where a provider had nothing to add.
#[inline]
pub(crate) fn gemm_provider() -> Option<&'static dyn GemmProvider> {
    if rayon::current_thread_index().is_some() {
        return None;
    }
    PROVIDER.get().map(|boxed| boxed.as_ref())
}

/// Offer one `f32` product; `false` means compute it here.
///
/// The two of these are what a call site uses. They read as one condition --
/// "unless somebody else did it" -- which is what a GEMM call site wants
/// wrapped around it, rather than three lines of plumbing repeated at each.
#[inline]
pub(crate) fn offer_gemm_f32(request: Gemm<'_, f32>) -> bool {
    gemm_provider().is_some_and(|provider| provider.gemm_f32(request))
}

/// Offer one `f64` product; see [`offer_gemm_f32`].
#[inline]
pub(crate) fn offer_gemm_f64(request: Gemm<'_, f64>) -> bool {
    gemm_provider().is_some_and(|provider| provider.gemm_f64(request))
}
