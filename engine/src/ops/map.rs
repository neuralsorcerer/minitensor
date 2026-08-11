// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Element-map primitives that produce fully-initialized output buffers.
//!
//! Every element-wise kernel used to fill a zero-initialized scratch buffer
//! (`TensorData::uninitialized_on_device`, which `memset`s for soundness) and
//! then overwrite every element — paying for two passes over the output. The
//! helpers here allocate the output as raw capacity (`Vec::with_capacity`),
//! write each element exactly once through `MaybeUninit`, and only then mark
//! the vector initialized, so the `memset` pass disappears without
//! reintroducing the undefined behavior the zeroing was added to fix.
//!
//! All `unsafe` involved in this pattern is confined to this module: the safe
//! combinators ([`unary_map`], [`binary_map`], …) initialize every element by
//! construction, and the one escape hatch for kernels with bespoke write
//! orders ([`build_vec_with`]) is an `unsafe fn` with an explicit contract.

use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::mem::MaybeUninit;

/// Element count above which *cheap* unary kernels switch to parallel
/// execution -- `relu`, `abs`, `sqrt`, `floor`, sign, casts, predicates.
/// Shared crate-wide (gradient kernels, activation maps, …).
///
/// Entering a rayon region costs a fixed ~25us here when the workers have
/// parked, which they do between calls from Python. A cheap unary op moves
/// about 0.05ns per element per core, so that overhead is not repaid until
/// the array is large. Measured on a 4-core x86-64 container, float32 `relu`:
///
/// ```text
///        N   sequential   parallel
///     4096       1.4 us    32.4 us   <- 23x slower parallel
///    16384       2.5 us    26.2 us
///    65536       8.6 us    26.4 us
///   262144      84.5 us    54.0 us   <- parallel finally wins
///  1048576     373.8 us   150.5 us
/// ```
///
/// The previous value of 4096 therefore made every cheap unary op between 4K
/// and ~200K elements slower than doing nothing at all, by up to 23x. This
/// value sits below the measured crossover on that machine so that hosts with
/// more cores -- where parallel repays sooner -- are not held back.
pub(crate) const PAR_THRESHOLD: usize = 1 << 17; // 131072 elements

/// Element count above which *expensive* unary kernels parallelize: the
/// transcendentals, whose per-element cost is hundreds of times a `relu`'s
/// (float32 `tanh` measures ~27ns per element per core against `relu`'s
/// ~0.05ns). The fixed region-entry cost is repaid almost immediately, so
/// these keep the low threshold, and parallel is a win from 4096 up:
///
/// ```text
///        N   sequential   parallel
///     4096      104 us      82 us    1.3x
///    65536     1626 us     559 us    2.9x
///  1048576    26114 us    7024 us    3.7x
/// ```
pub(crate) const EXPENSIVE_PAR_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Element count above which the vectorized float32 kernels in
/// `ops::simd::transcendental` parallelize -- `tanh`, `erf` and both GELU
/// variants.
///
/// They get their own threshold because they are no longer expensive kernels.
/// [`EXPENSIVE_PAR_THRESHOLD`] is calibrated for transcendentals costing tens of
/// nanoseconds per element; the vectorized `tanh` in `ops::simd::transcendental`
/// cost 2 to 3, so the fixed region-entry cost takes an order of magnitude
/// more elements to repay. Fitting both sides for `tanh` on a 4-core machine:
///
/// ```text
///   sequential   2.17 ns/elem +  0.0 us fixed
///   parallel     0.66 ns/elem + 26.1 us fixed   -> they cross at N ~ 18500
/// ```
///
/// Same convention as [`PAR_THRESHOLD`]: sit just below the measured crossover,
/// so hosts with more cores -- where the parallel side is cheaper per element
/// and repays sooner -- are not held back.
pub(crate) const VECTOR_F32_PAR_THRESHOLD: usize = 1 << 14; // 16384 elements

/// Element count above which binary/broadcast kernels parallelize.
///
/// Same reasoning as [`PAR_THRESHOLD`]. Note this only governs the
/// *broadcasting* path: equal-shape elementwise binary ops take the sequential
/// SIMD fast path in `ops::kernels::binary` and never reach rayon at all.
///
/// The old value of 1024 is one `PAR_CHUNK`, so it was the first size at which
/// a split actually happens -- and therefore the first size to pay the
/// worker-wake cost. Measured broadcast add (`Nx1 + 1xN`, float32) on a
/// 4-core machine:
///
/// ```text
///        N   minitensor
///     1024       1.4 us   (one chunk: runs inline, no wake)
///     4096      21.6 us   <- waking the workers costs ~20us flat
///    16384      25.6 us
///    65536      30.9 us   <- enough work to amortize the wake
/// ```
pub(crate) const BINARY_PAR_THRESHOLD: usize = 1 << 15; // 32768 elements

/// Chunk size for parallel map loops.
pub(crate) const PAR_CHUNK: usize = 1024;

/// Build a `Vec<U>` of exactly `len` elements, delegating initialization of
/// the spare capacity to `fill`.
///
/// # Safety
///
/// When `fill` returns `Ok(())` it must have initialized **every** element of
/// the slice it was given. If `fill` returns `Err`, the partially written
/// buffer is discarded without being marked initialized, so error paths are
/// safe regardless.
pub(crate) unsafe fn build_vec_with<U, E, F>(len: usize, fill: F) -> Result<Vec<U>, E>
where
    F: FnOnce(&mut [MaybeUninit<U>]) -> Result<(), E>,
{
    let mut out: Vec<U> = Vec::with_capacity(len);
    fill(&mut out.spare_capacity_mut()[..len])?;
    // SAFETY: `fill` returned Ok, so per this function's contract all `len`
    // elements are initialized.
    unsafe { out.set_len(len) };
    Ok(out)
}

/// Sequential core: write `op(input[i])` into every element of `out`.
#[inline(always)]
fn map_into<T, U, F>(input: &[T], out: &mut [MaybeUninit<U>], op: &F)
where
    T: Copy,
    F: Fn(T) -> U,
{
    debug_assert_eq!(input.len(), out.len());
    for (o, &i) in out.iter_mut().zip(input.iter()) {
        o.write(op(i));
    }
}

/// Sequential core: write `op(lhs[i], rhs[i])` into every element of `out`.
#[inline(always)]
fn zip_into<A, B, U, F>(lhs: &[A], rhs: &[B], out: &mut [MaybeUninit<U>], op: &F)
where
    A: Copy,
    B: Copy,
    F: Fn(A, B) -> U,
{
    debug_assert_eq!(lhs.len(), out.len());
    debug_assert_eq!(rhs.len(), out.len());
    for ((o, &l), &r) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        o.write(op(l, r));
    }
}

/// Sequential core: write `op(a[i], b[i], c[i])` into every element of `out`.
#[inline(always)]
fn zip3_into<A, B, C, U, F>(a: &[A], b: &[B], c: &[C], out: &mut [MaybeUninit<U>], op: &F)
where
    A: Copy,
    B: Copy,
    C: Copy,
    F: Fn(A, B, C) -> U,
{
    debug_assert_eq!(a.len(), out.len());
    debug_assert_eq!(b.len(), out.len());
    debug_assert_eq!(c.len(), out.len());
    for (((o, &x), &y), &z) in out.iter_mut().zip(a.iter()).zip(b.iter()).zip(c.iter()) {
        o.write(op(x, y, z));
    }
}

/// Map `op` over `input` into a fresh, exactly-sized `Vec` (no zeroing pass).
/// Parallel above `threshold`.
pub(crate) fn unary_map_threshold<T, U, F>(input: &[T], threshold: usize, op: F) -> Vec<U>
where
    T: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(T) -> U + Send + Sync,
{
    let len = input.len();
    // SAFETY: both branches write every element of the spare slice —
    // `map_into` walks the full zip of equal-length slices, and the parallel
    // chunk split covers the output exactly.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < threshold {
                map_into(input, spare, &op);
            } else {
                input
                    .par_chunks(PAR_CHUNK)
                    .zip(spare.par_chunks_mut(PAR_CHUNK))
                    .for_each(|(ic, oc)| map_into(ic, oc, &op));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

/// [`unary_map_threshold`], but handing `op` a whole contiguous block at a time
/// instead of one element at a time.
///
/// A per-element `Fn(T) -> U` is the wrong shape for a kernel that wants to be
/// vectorized: the closure is opaque at the call site, so the loop that drives
/// it cannot be turned into vector code. Kernels that carry their own
/// `#[target_feature]` instantiations (see `crate::ops::simd::transcendental`)
/// need the loop *inside* the multiversioned function, which means being handed
/// the slice.
///
/// # Safety
///
/// On return `op` must have initialized **every** element of each output block
/// it was given. The blocking here covers the output exactly, so initializing
/// each block in full initializes the whole `Vec`.
pub(crate) unsafe fn unary_map_blocks_threshold<T, U, F>(
    input: &[T],
    threshold: usize,
    op: F,
) -> Vec<U>
where
    T: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(&[T], &mut [MaybeUninit<U>]) + Send + Sync,
{
    let len = input.len();
    // SAFETY: forwarded to the caller by this function's own contract — `op`
    // initializes every element of every block, and the blocks tile the output.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < threshold {
                op(input, spare);
            } else {
                input
                    .par_chunks(PAR_CHUNK)
                    .zip(spare.par_chunks_mut(PAR_CHUNK))
                    .for_each(|(ic, oc)| op(ic, oc));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

/// [`unary_map_threshold`] at the crate-wide unary threshold.
#[inline]
pub(crate) fn unary_map<T, U, F>(input: &[T], op: F) -> Vec<U>
where
    T: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(T) -> U + Send + Sync,
{
    unary_map_threshold(input, PAR_THRESHOLD, op)
}

/// Write `op(input[i])` into an existing output slice, parallel above
/// [`PAR_THRESHOLD`].
///
/// The in-place counterpart to [`unary_map`], for kernels that already own a
/// destination buffer. Chunked rather than indexed, so the bounds stay visible
/// to the optimizer and no pointer has to be laundered across the rayon
/// closure boundary.
pub(crate) fn unary_map_into<T, U, F>(out: &mut [U], input: &[T], op: F)
where
    T: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(T) -> U + Send + Sync,
{
    debug_assert_eq!(out.len(), input.len());
    let apply = |out: &mut [U], input: &[T]| {
        for (o, &i) in out.iter_mut().zip(input.iter()) {
            *o = op(i);
        }
    };
    if out.len() < PAR_THRESHOLD {
        apply(out, input);
    } else {
        out.par_chunks_mut(PAR_CHUNK)
            .zip(input.par_chunks(PAR_CHUNK))
            .for_each(|(o, i)| apply(o, i));
    }
}

/// Zip `op` over two equal-length slices into a fresh, exactly-sized `Vec`.
/// Parallel above [`BINARY_PAR_THRESHOLD`]. The two inputs may have different
/// element types (e.g. zipping values with a boolean mask).
pub(crate) fn binary_map<A, B, U, F>(lhs: &[A], rhs: &[B], op: F) -> Vec<U>
where
    A: Copy + Sync,
    B: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(A, B) -> U + Send + Sync,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    let len = lhs.len();
    // SAFETY: both branches write every element of the spare slice.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < BINARY_PAR_THRESHOLD {
                zip_into(lhs, rhs, spare, &op);
            } else {
                lhs.par_chunks(PAR_CHUNK)
                    .zip(rhs.par_chunks(PAR_CHUNK))
                    .zip(spare.par_chunks_mut(PAR_CHUNK))
                    .for_each(|((lc, rc), oc)| zip_into(lc, rc, oc, &op));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

/// [`binary_map`], but handing `op` a whole contiguous block at a time.
///
/// The two-input counterpart to [`unary_map_blocks_threshold`], and there for
/// the same reason: a gradient kernel that carries its own `#[target_feature]`
/// instantiations needs the loop *inside* the multiversioned function, which
/// means being handed slices rather than elements. Gradient kernels take the
/// saved input and the incoming gradient, so they need two.
///
/// # Safety
///
/// On return `op` must have initialized **every** element of each output block
/// it was given. The blocking covers the output exactly, so initializing each
/// block in full initializes the whole `Vec`.
pub(crate) unsafe fn binary_map_blocks_threshold<A, B, U, F>(
    lhs: &[A],
    rhs: &[B],
    threshold: usize,
    op: F,
) -> Vec<U>
where
    A: Copy + Sync,
    B: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(&[A], &[B], &mut [MaybeUninit<U>]) + Send + Sync,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    let len = lhs.len();
    // SAFETY: forwarded to the caller by this function's own contract.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < threshold {
                op(lhs, rhs, spare);
            } else {
                lhs.par_chunks(PAR_CHUNK)
                    .zip(rhs.par_chunks(PAR_CHUNK))
                    .zip(spare.par_chunks_mut(PAR_CHUNK))
                    .for_each(|((lc, rc), oc)| op(lc, rc, oc));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

/// Zip `op` over three equal-length slices into a fresh, exactly-sized `Vec`.
/// Parallel above [`BINARY_PAR_THRESHOLD`].
///
/// Gradient kernels routinely combine three operands (saved input, saved
/// output, incoming gradient); expressing that here keeps them on the
/// write-once output path instead of a zero-then-overwrite buffer.
pub(crate) fn ternary_map<A, B, C, U, F>(a: &[A], b: &[B], c: &[C], op: F) -> Vec<U>
where
    A: Copy + Sync,
    B: Copy + Sync,
    C: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(A, B, C) -> U + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    let len = a.len();
    // SAFETY: both branches write every element of the spare slice.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < BINARY_PAR_THRESHOLD {
                zip3_into(a, b, c, spare, &op);
            } else {
                a.par_chunks(PAR_CHUNK)
                    .zip(b.par_chunks(PAR_CHUNK))
                    .zip(c.par_chunks(PAR_CHUNK))
                    .zip(spare.par_chunks_mut(PAR_CHUNK))
                    .for_each(|(((ac, bc), cc), oc)| zip3_into(ac, bc, cc, oc, &op));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

/// Gather a strided view into a fresh contiguous (row-major) buffer.
///
/// `dims`/`strides` describe the source view (element strides; stride 0 is
/// valid and repeats the element, as `expand` produces). Every output element
/// is written exactly once, so no zeroing pass is needed. Out-of-bounds
/// views panic via safe indexing rather than reading out of bounds.
///
/// Replaces the previous `copy_strided_to_contiguous`, which was fully
/// sequential and recomputed the source offset from scratch for every
/// element; this walker maintains a running offset and parallelizes above
/// [`PAR_THRESHOLD`].
pub(crate) fn strided_gather<T: Copy + Send + Sync>(
    src: &[T],
    dims: &[usize],
    strides: &[usize],
) -> Vec<T> {
    debug_assert_eq!(dims.len(), strides.len());
    if dims.is_empty() {
        return vec![src[0]];
    }
    let numel: usize = dims.iter().product();
    if numel == 0 {
        return Vec::new();
    }
    let rank = dims.len();

    let walk = |start: usize, chunk: &mut [MaybeUninit<T>]| {
        let mut index: SmallVec<[usize; 8]> = smallvec![0; rank];
        let mut offset = 0usize;
        let mut tmp = start;
        for i in (0..rank).rev() {
            index[i] = tmp % dims[i];
            tmp /= dims[i];
            offset += index[i] * strides[i];
        }
        for o in chunk.iter_mut() {
            o.write(src[offset]);
            for dim in (0..rank).rev() {
                index[dim] += 1;
                offset += strides[dim];
                if index[dim] < dims[dim] {
                    break;
                }
                index[dim] = 0;
                offset -= strides[dim] * dims[dim];
            }
        }
    };

    // SAFETY: both paths write every element of the spare slice (the chunks
    // partition it exactly).
    unsafe {
        build_vec_with::<T, std::convert::Infallible, _>(numel, |spare| {
            if numel < PAR_THRESHOLD {
                walk(0, spare);
            } else {
                spare
                    .par_chunks_mut(PAR_CHUNK)
                    .enumerate()
                    .for_each(|(ci, chunk)| walk(ci * PAR_CHUNK, chunk));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_map_matches_reference_sequential_and_parallel() {
        for len in [0usize, 1, 7, 1023, 1024, 4096, 10_000] {
            let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let expected: Vec<f32> = input.iter().map(|x| x * 2.0 + 1.0).collect();
            assert_eq!(unary_map(&input, |x: f32| x * 2.0 + 1.0), expected, "{len}");
        }
    }

    #[test]
    fn unary_map_supports_type_changing_ops() {
        let input = vec![1.5f64, -2.0, 0.0];
        let out: Vec<bool> = unary_map(&input, |x: f64| x > 0.0);
        assert_eq!(out, vec![true, false, false]);
    }

    #[test]
    fn binary_map_matches_reference_sequential_and_parallel() {
        for len in [0usize, 1, 5, 1023, 1024, 4097, 10_000] {
            let a: Vec<i64> = (0..len).map(|i| i as i64).collect();
            let b: Vec<i64> = (0..len).map(|i| (i * 3) as i64).collect();
            let expected: Vec<i64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            assert_eq!(
                binary_map(&a, &b, |x: i64, y: i64| x + y),
                expected,
                "{len}"
            );
        }
    }

    #[test]
    fn strided_gather_handles_views_and_scalars() {
        // 2x3 row-major identity gather
        let src = [1, 2, 3, 4, 5, 6];
        assert_eq!(
            strided_gather(&src, &[2, 3], &[3, 1]),
            vec![1, 2, 3, 4, 5, 6]
        );
        // transpose view: dims [3,2], strides [1,3]
        assert_eq!(
            strided_gather(&src, &[3, 2], &[1, 3]),
            vec![1, 4, 2, 5, 3, 6]
        );
        // broadcast (stride 0) view: one row repeated
        assert_eq!(
            strided_gather(&src[..3], &[2, 3], &[0, 1]),
            vec![1, 2, 3, 1, 2, 3]
        );
        // 0-d
        assert_eq!(strided_gather(&src, &[], &[]), vec![1]);
        // empty
        assert_eq!(strided_gather(&src, &[0, 3], &[3, 1]), Vec::<i32>::new());
        // parallel path matches sequential reference
        let big: Vec<i64> = (0..10_000).collect();
        let gathered = strided_gather(&big, &[100, 100], &[1, 100]);
        for r in 0..100 {
            for c in 0..100 {
                assert_eq!(gathered[r * 100 + c], big[c * 100 + r]);
            }
        }
    }

    #[test]
    fn build_vec_with_error_discards_buffer_safely() {
        let result: Result<Vec<f32>, &str> = unsafe {
            build_vec_with(16, |spare| {
                // Partially initialize, then fail: must not leak or UB.
                spare[0].write(1.0);
                Err("boom")
            })
        };
        assert_eq!(result.unwrap_err(), "boom");
    }
}
