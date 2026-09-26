// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! A seam for handing work to a library that does it faster than the engine.
//!
//! The engine's kernels are the right thing to run almost everywhere, and the
//! exceptions are specific and measured: a single large dense product in
//! `float32` or `float64`, where a tuned BLAS wins by 1.6-2.3x, and the float64
//! transcendentals, where the engine calls scalar `libm` and NumPy runs
//! vectorized loops that are both faster and at least as accurate.
//!
//! The library already ships next to both: `numpy`, which every install pulls
//! in, carries a full OpenBLAS and those loops. Reaching them costs nothing but
//! array headers pointed at buffers the engine already owns. That is what a
//! provider is -- the Python bindings install one at import, and a build with
//! no Python in it (`cargo test`, an embedding Rust program) simply has none
//! and runs the engine's kernels, which is why this is a hook and not a
//! dependency.
//!
//! The engine offers the *whole* problem and the provider decides. It sees the
//! operation and its size before it commits, so work too small to be worth a
//! call, or an operation the engine does better, declines and costs one
//! predictable branch. The table of what is worth sending lives with the
//! provider, next to the measurements it came from.

use std::mem::MaybeUninit;
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

/// One dense product, or `batch` of them: `out[b] = lhs[b] @ rhs[b]`.
///
/// `lhs` is `[batch, m, k]`, `rhs` is `[batch, k, n]` and `out` is
/// `[batch, m, n]`, each packed with no padding -- the engine materialises
/// every view, so nothing here is strided beyond the two matrix axes being in
/// one order or the other. `out` is row-major, arrives zeroed, and is written
/// rather than accumulated into.
///
/// A `batch` of 1 is the ordinary single product and is what most call sites
/// build. The batched form exists because handing a stack of matrices over one
/// at a time spends the crossing cost per matrix: at a batch of 256 that is
/// more than the whole product costs to compute. One request, one crossing.
pub struct Gemm<'a, T> {
    /// Independent products laid out one after another. Never zero.
    pub batch: usize,
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
    /// `batch * m * n * k` rather than the dimensions separately: `1x1024 @
    /// 1024x1` and `1024x1 @ 1x1024` are the same three numbers and nothing
    /// alike as work, and only the product tells them apart.
    pub fn flops(&self) -> usize {
        self.batch
            .saturating_mul(self.m)
            .saturating_mul(self.n)
            .saturating_mul(self.k)
    }
}

/// A float64 element-wise function the engine will offer to a provider.
///
/// Only those a provider might do better are named: offering is a call and a
/// match, and there is no point paying either for a function the engine
/// always keeps. What *is* worth sending, and from which size, is the
/// provider's decision. `Atan2` and `Pow` take two operands and are offered
/// through [`Provider::binary_f64`]; the rest take one.
///
/// Float64 only. Every float32 kernel in `ops::simd::transcendental` computes
/// in float64 and rounds once, which makes it correctly rounded on every
/// input; no library that answers faster answers as well.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ufunc {
    Tanh,
    Sinh,
    Cosh,
    Tan,
    Asinh,
    Atanh,
    Expm1,
    Log1p,
    Log2,
    Log10,
    Exp,
    Exp2,
    Log,
    Cbrt,
    Asin,
    Acos,
    Atan,
    Acosh,
    Atan2,
    Pow,
}

impl Ufunc {
    /// The NumPy ufunc of the same function.
    pub fn numpy_name(self) -> &'static str {
        match self {
            Ufunc::Tanh => "tanh",
            Ufunc::Sinh => "sinh",
            Ufunc::Cosh => "cosh",
            Ufunc::Tan => "tan",
            Ufunc::Asinh => "arcsinh",
            Ufunc::Atanh => "arctanh",
            Ufunc::Expm1 => "expm1",
            Ufunc::Log1p => "log1p",
            Ufunc::Log2 => "log2",
            Ufunc::Log10 => "log10",
            Ufunc::Exp => "exp",
            Ufunc::Exp2 => "exp2",
            Ufunc::Log => "log",
            Ufunc::Cbrt => "cbrt",
            Ufunc::Asin => "arcsin",
            Ufunc::Acos => "arccos",
            Ufunc::Atan => "arctan",
            Ufunc::Acosh => "arccosh",
            Ufunc::Atan2 => "arctan2",
            Ufunc::Pow => "power",
        }
    }
}

/// Somewhere better to send work than the engine's own kernel.
///
/// Each method answers whether it did the work. `false` means the engine runs
/// its own, so declining is always safe and always correct; a provider that
/// cannot help with a particular size or dtype says so rather than arranging
/// to be asked less often.
///
/// A provider must leave a GEMM's `out` exactly as it found it -- zeroed --
/// when it declines or fails part way, since the fallback writes assuming
/// that. An element-wise `out` arrives uninitialized, and is read only if the
/// provider says it wrote every element of it.
pub trait Provider: Send + Sync {
    fn gemm_f32(&self, request: Gemm<'_, f32>) -> bool;
    fn gemm_f64(&self, request: Gemm<'_, f64>) -> bool;

    /// `out[i] = op(input[i])` for every `i`, or `false` having promised
    /// nothing about `out`. The two slices are the same length.
    fn unary_f64(&self, _op: Ufunc, _input: &[f64], _out: &mut [MaybeUninit<f64>]) -> bool {
        false
    }

    /// `out[i] = op(lhs[i], rhs[i])` for every `i`, or `false` having promised
    /// nothing about `out`. An operand is either `out`'s length or a single
    /// element that stands for every position, as `x ** 2.5` has one.
    fn binary_f64(
        &self,
        _op: Ufunc,
        _lhs: &[f64],
        _rhs: &[f64],
        _out: &mut [MaybeUninit<f64>],
    ) -> bool {
        false
    }
}

static PROVIDER: OnceLock<Box<dyn Provider>> = OnceLock::new();

/// Install the provider for the life of the process.
///
/// Returns whether this call was the one that installed it. Once, because a
/// provider is a property of the host rather than of a call -- the bindings
/// install theirs while the extension module is initialising, before any
/// tensor exists -- and because a swap would have to be synchronised against
/// every in-flight GEMM to be sound, at the cost of a lock on the path this
/// exists to make faster.
pub fn set_provider(provider: Box<dyn Provider>) -> bool {
    PROVIDER.set(provider).is_ok()
}

/// Whether a provider is installed at all, ignoring the thread restriction.
///
/// For introspection -- "will this build delegate?" -- rather than for
/// dispatch, which is what [`provider`] is for and which additionally
/// depends on where it is asked from.
pub fn provider_installed() -> bool {
    PROVIDER.get().is_some()
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
pub(crate) fn provider() -> Option<&'static dyn Provider> {
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
    provider().is_some_and(|provider| provider.gemm_f32(request))
}

/// Offer one `f64` product; see [`offer_gemm_f32`].
#[inline]
pub(crate) fn offer_gemm_f64(request: Gemm<'_, f64>) -> bool {
    provider().is_some_and(|provider| provider.gemm_f64(request))
}

/// How many threads a provider may spread one piece of work over: the size of
/// the engine's own pool, so handing work over never changes how much of the
/// machine it is allowed.
pub fn pool_threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// `op` over `input` from the provider, or `None` to compute it here.
#[inline]
pub(crate) fn offer_unary_f64(op: Ufunc, input: &[f64]) -> Option<Vec<f64>> {
    let provider = provider()?;
    collect(input.len(), |out| provider.unary_f64(op, input, out))
}

/// `op` over `lhs` and `rhs` from the provider, or `None` to compute it here.
/// Each operand is `len` long or a single element; see
/// [`Provider::binary_f64`].
#[inline]
pub(crate) fn offer_binary_f64(
    op: Ufunc,
    lhs: &[f64],
    rhs: &[f64],
    len: usize,
) -> Option<Vec<f64>> {
    let provider = provider()?;
    collect(len, |out| provider.binary_f64(op, lhs, rhs, out))
}

/// A fresh `len`-element buffer, if `fill` says it wrote every element.
fn collect(len: usize, fill: impl FnOnce(&mut [MaybeUninit<f64>]) -> bool) -> Option<Vec<f64>> {
    let mut out: Vec<f64> = Vec::with_capacity(len);
    if !fill(&mut out.spare_capacity_mut()[..len]) {
        return None;
    }
    // SAFETY: the provider said it wrote every one of the `len` elements,
    // which is its contract for `true`.
    unsafe { out.set_len(len) };
    Some(out)
}
