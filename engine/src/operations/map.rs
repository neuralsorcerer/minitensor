// Copyright (c) 2026 Soumyadip Sarkar.
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

/// Element count above which unary kernels switch to parallel execution.
/// Shared crate-wide (gradient kernels, activation maps, …).
pub(crate) const PAR_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Element count above which binary/broadcast kernels parallelize. Kept at
/// the historical `broadcast_binary_op` threshold so parallelization behavior
/// is unchanged by the uninit-output refactor.
pub(crate) const BINARY_PAR_THRESHOLD: usize = 1024;

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
fn zip_into<T, U, F>(lhs: &[T], rhs: &[T], out: &mut [MaybeUninit<U>], op: &F)
where
    T: Copy,
    F: Fn(T, T) -> U,
{
    debug_assert_eq!(lhs.len(), out.len());
    debug_assert_eq!(rhs.len(), out.len());
    for ((o, &l), &r) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        o.write(op(l, r));
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

/// Zip `op` over two equal-length slices into a fresh, exactly-sized `Vec`.
/// Parallel above [`BINARY_PAR_THRESHOLD`].
pub(crate) fn binary_map<T, U, F>(lhs: &[T], rhs: &[T], op: F) -> Vec<U>
where
    T: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(T, T) -> U + Send + Sync,
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
