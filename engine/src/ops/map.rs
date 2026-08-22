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
/// Same reasoning as [`PAR_THRESHOLD`]. Note this governs the *broadcasting*
/// path, and the dtypes with no vectorized kernel; equal-shape f32/f64
/// arithmetic takes the fast path in `ops::kernels::binary` and splits at
/// [`SIMD_PAR_THRESHOLD`] instead, which is measured separately because the
/// work per element is smaller there.
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

/// Element count above which the equal-shape arithmetic kernels in
/// `ops::kernels::binary` parallelize — the `+`, `-`, `*`, `/` that every other
/// operation is built out of.
///
/// They get their own threshold for the same reason
/// [`VECTOR_F32_PAR_THRESHOLD`] does, one step further along: at one arithmetic
/// operation per two loads and a store, they are the cheapest kernels in the
/// engine, so the fixed region-entry cost takes the most elements to repay.
/// Measured on a 4-core x86-64 container, float32 `add` (us):
///
/// ```text
///        N   sequential   parallel
///    16384         2.34       6.25
///    32768         4.80       7.41
///    65536        10.52       9.64   <- parallel takes the lead
///   131072        48.37      15.71     3.1x
///  1048576       506.49     124.13     4.1x
///  4194304      4200.93     855.13     4.9x
/// ```
///
/// Sitting *below* the crossover is the usual convention (see
/// [`PAR_THRESHOLD`]), and the reason to sit only just below it is the middle
/// row: the sequential side is still winning at 32768.
///
/// Note the shape of the sequential column — it is linear to 65536 and then
/// steps by 4.6x at 131072, where the three buffers stop fitting in cache. Past
/// that point one core is waiting on memory, and the parallel speedup is mostly
/// the other three cores' load/store units rather than their arithmetic.
pub(crate) const SIMD_PAR_THRESHOLD: usize = 1 << 16; // 65536 elements

/// Chunk size for parallel map loops.
pub(crate) const PAR_CHUNK: usize = 1024;

/// Chunk size for the arithmetic kernels' parallel loops.
///
/// Eight times [`PAR_CHUNK`], because these kernels are cheap enough per
/// element that a 1024-element block spends a visible fraction of itself
/// entering and leaving. Same measurement as [`SIMD_PAR_THRESHOLD`], varying
/// only the block length (float32 `add`, us):
///
/// ```text
///        N   1024-elem   8192-elem
///    65536       13.64        9.64
///   131072       17.76       15.71
///   262144       41.59       33.58
///  4194304      925.70      855.13
/// ```
///
/// Above 8192 the blocks start to outrun the cache again (at 65536 elements per
/// block, 262144 measured 28.66us but 4194304 rose to 1044.57), so this is the
/// length that is good everywhere rather than best anywhere.
pub(crate) const SIMD_PAR_CHUNK: usize = 8192;

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

/// Build a `Vec<U>` of exactly `len` elements, handing the raw capacity to
/// `fill`, which must write all of it.
///
/// The infallible form of [`build_vec_with`], and the one the data-movement
/// kernels want. Those relocate elements rather than computing them, so they
/// have no `Result` to thread through, but they were paying the cost this
/// module exists to remove all the same: `vec![T::default(); n]` zeroes the
/// whole output and then `copy_from_slice` overwrites every byte of it. Two
/// passes to move data once, which on a concatenation of two million-element
/// float32 arrays was most of the difference against NumPy.
///
/// # Safety
///
/// `fill` must initialize **every** element of the slice it is given.
pub(crate) unsafe fn build_vec<U, F>(len: usize, fill: F) -> Vec<U>
where
    F: FnOnce(&mut [MaybeUninit<U>]),
{
    // SAFETY: forwarded to the caller by this function's own contract.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            fill(spare);
            Ok(())
        })
    }
    .unwrap_or_else(|e| match e {})
}

/// The type-erased body of a one-input parallel map: one input chunk in, one
/// output chunk out. Named so the `&dyn` signatures below stay readable.
type ChunkWork<'a, T, U> = &'a (dyn Fn(&[T], &mut [MaybeUninit<U>]) + Sync);

/// [`ChunkWork`] for two inputs.
type ChunkWork2<'a, A, B, U> = &'a (dyn Fn(&[A], &[B], &mut [MaybeUninit<U>]) + Sync);

/// [`ChunkWork`] for three.
type ChunkWork3<'a, A, B, C, U> = &'a (dyn Fn(&[A], &[B], &[C], &mut [MaybeUninit<U>]) + Sync);

/// Drive `work` over matching chunks of one input and the output, with the
/// closure **type-erased**.
///
/// This is the one place the parallel split for element maps happens, and it
/// takes `&dyn Fn` rather than a generic closure on purpose. Rayon's iterator
/// plumbing — `StackJob`, `join_context`, the bridge — is deeply generic, so it
/// is instantiated afresh for every distinct closure type handed to a
/// `par_chunks` pipeline. With one instantiation per call site across the
/// engine, that machinery was 4.4 MB of a 12.7 MB extension module, 42% of the
/// shipped binary, for kernels whose own loops are a few hundred bytes each.
///
/// Erasing the closure collapses that to one instantiation per element-type
/// pair. What it costs is an indirect call per *chunk* — not per element — so
/// over a `PAR_CHUNK` of a thousand elements it is beneath measurement, and the
/// loop inside `work` is still fully inlined and vectorized because that
/// inlining happens on the other side of the boundary.
fn par_zip_chunks<T, U>(
    input: &[T],
    out: &mut [MaybeUninit<U>],
    chunk: usize,
    work: ChunkWork<T, U>,
) where
    T: Sync,
    U: Send + Sync,
{
    input
        .par_chunks(chunk)
        .zip(out.par_chunks_mut(chunk))
        .for_each(|(input_chunk, out_chunk)| work(input_chunk, out_chunk));
}

/// [`par_zip_chunks`] for the two-input maps.
fn par_zip_chunks2<A, B, U>(
    lhs: &[A],
    rhs: &[B],
    out: &mut [MaybeUninit<U>],
    chunk: usize,
    work: ChunkWork2<A, B, U>,
) where
    A: Sync,
    B: Sync,
    U: Send + Sync,
{
    lhs.par_chunks(chunk)
        .zip(rhs.par_chunks(chunk))
        .zip(out.par_chunks_mut(chunk))
        .for_each(|((lhs_chunk, rhs_chunk), out_chunk)| work(lhs_chunk, rhs_chunk, out_chunk));
}

/// [`par_zip_chunks`] for the three-input maps.
fn par_zip_chunks3<A, B, C, U>(
    a: &[A],
    b: &[B],
    c: &[C],
    out: &mut [MaybeUninit<U>],
    chunk: usize,
    work: ChunkWork3<A, B, C, U>,
) where
    A: Sync,
    B: Sync,
    C: Sync,
    U: Send + Sync,
{
    a.par_chunks(chunk)
        .zip(b.par_chunks(chunk))
        .zip(c.par_chunks(chunk))
        .zip(out.par_chunks_mut(chunk))
        .for_each(|(((ac, bc), cc), oc)| work(ac, bc, cc, oc));
}

/// How many outputs to give one parallel task, when producing each output costs
/// `width` element reads.
///
/// Reduction kernels vary enormously in how much work one output is: a row sum
/// over a 4096-wide matrix is thousands of reads, a reduction over a length-2
/// axis is two. A fixed chunk width is wrong for one end or the other, so scale
/// it to hold the *work* per task roughly constant instead.
#[inline]
pub(crate) fn outputs_per_task(width: usize) -> usize {
    /// Element reads per task. Large enough to bury the split bookkeeping,
    /// small enough that a few thousand outputs still fill every core.
    const TARGET: usize = 1 << 14;
    (TARGET / width.max(1)).max(1)
}

/// The type-erased body of an output-partitioned parallel loop: the index of
/// the chunk's first output element, and the chunk itself.
type OutWork<'a, T> = &'a (dyn Fn(usize, &mut [T]) + Sync);

/// [`OutWork`] for kernels that fill two outputs in step — values and indices,
/// as `sort`, `topk` and the quantile kernels do.
type OutWork2<'a, T, U> = &'a (dyn Fn(usize, &mut [T], &mut [U]) + Sync);

/// Split `out` into contiguous chunks of `chunk` elements and run `work` on
/// each in parallel, passing the index of the chunk's first element.
///
/// This is the reduction-shaped sibling of [`par_zip_chunks`], and it is erased
/// for the same reason: a `par_chunks_mut(..).enumerate().for_each(..)` pipeline
/// instantiates rayon's splitter, `StackJob` and bridge afresh for every
/// distinct closure type, and the engine writes that pipeline by hand at over
/// two hundred sites. Erased, they share one instantiation per output element
/// type.
///
/// It also fixes a granularity bug the hand-written form kept making. Many of
/// those sites were `out.par_iter_mut().enumerate()`, which hands rayon *one
/// work item per output element* — for a reduction whose per-element body is a
/// short strided walk, the split bookkeeping can cost more than the arithmetic.
/// Here the unit of work is a chunk, and the caller picks its width.
///
/// Every output chunk is computed independently of the others, so the partition
/// cannot affect the result: this is safe to use in kernels that must stay
/// bitwise stable across thread counts, and unsafe to use where the partition
/// decides how values are *grouped* into an accumulation (see
/// `reduce_along_dim0`, which fixes its row bands to constants for exactly that
/// reason).
pub(crate) fn par_out_chunks<T: Send>(out: &mut [T], chunk: usize, work: OutWork<T>) {
    // No output means no chunks and so no calls, matching `par_chunks_mut`
    // exactly. This is load-bearing: several kernels have a `work` that assumes
    // its chunk is non-empty (it indexes the first row), and would panic rather
    // than do nothing on a zero-sized tensor. It also covers `chunk == 0`,
    // which only arises when some axis inside the reduced one is empty — and
    // that empties the output too.
    if out.is_empty() {
        return;
    }
    // A single chunk is the common small-tensor case; running it here skips the
    // rayon dispatch entirely.
    if out.len() <= chunk || chunk == 0 {
        work(0, out);
        return;
    }
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(index, out_chunk)| work(index * chunk, out_chunk));
}

/// The state buffers one optimizer step writes, split to match a parameter
/// chunk. Four is past every optimizer in the engine (Adam's widest is `m`,
/// `v`, `v_hat`), so the split never reaches the heap.
type StateChunks<'a, T> = SmallVec<[&'a mut [T]; 4]>;

/// The type-erased body of a row-partitioned kernel: the first row's index and
/// one window per output buffer.
type RowWork<'a, T> = &'a (dyn Fn(usize, &mut [&mut [T]]) + Sync);

/// [`RowWork`] for an optimizer step, which also gets its gradient window.
type UpdateWork<'a, T> = &'a (dyn Fn(&mut [T], &[T], &mut [&mut [T]]) + Sync);

/// [`OutWork`] for chunk work that can fail.
type TryOutWork<'a, T, E> = &'a (dyn Fn(usize, &mut [T]) -> Result<(), E> + Sync);

/// Split `rows` rows of work across threads, cutting **several output buffers
/// at the same row boundary** — each with its own number of elements per row.
///
/// This is the shape the zipped pipelines could not express. A layer norm
/// writes three things per row: `norm` normalized values, `norm` scaled-and-
/// shifted outputs, and one reciprocal standard deviation. Zipping them means a
/// four-deep `Zip` of `par_chunks_mut(norm)`, `par_chunks_mut(norm)`,
/// `par_iter_mut()` and `par_chunks(norm)`, written once per dtype per variant,
/// because a `Zip` needs every side to yield the same number of items and these
/// have different widths. Here the widths are data: buffer `i` gives each row
/// `widths[i]` elements, and the split divides all of them at the same row.
///
/// Inputs do not appear at all. They are shared references, so `work` captures
/// them and indexes from the starting row — which is what removed the need for
/// a driver taking both a mutable and an immutable buffer set.
///
/// `work` receives the first row's index and one window per buffer. Rows are
/// independent by construction, so the partition cannot change the result.
pub(crate) fn par_row_outputs<T: Send + Sync>(
    rows: usize,
    row_chunk: usize,
    outputs: &mut [&mut [T]],
    widths: &[usize],
    work: RowWork<T>,
) {
    debug_assert_eq!(outputs.len(), widths.len());
    debug_assert!(
        outputs
            .iter()
            .zip(widths)
            .all(|(buffer, &width)| buffer.len() == rows * width)
    );
    if rows == 0 {
        return;
    }
    let owned: StateChunks<T> = outputs.iter_mut().map(|buffer| &mut **buffer).collect();
    par_row_outputs_recurse(0, rows, row_chunk.max(1), owned, widths, work);
}

fn par_row_outputs_recurse<T: Send + Sync>(
    first_row: usize,
    rows: usize,
    row_chunk: usize,
    mut outputs: StateChunks<'_, T>,
    widths: &[usize],
    work: RowWork<T>,
) {
    if rows <= row_chunk {
        work(first_row, &mut outputs);
        return;
    }
    // Halve, rounded to a chunk boundary, so every leaf holds whole chunks and
    // `work` sees the same windows the hand-written pipelines passed.
    let mid = (rows / 2 / row_chunk).max(1) * row_chunk;
    let mut lo: StateChunks<T> = SmallVec::with_capacity(outputs.len());
    let mut hi: StateChunks<T> = SmallVec::with_capacity(outputs.len());
    for (buffer, &width) in outputs.into_iter().zip(widths) {
        let (buffer_lo, buffer_hi) = buffer.split_at_mut(mid * width);
        lo.push(buffer_lo);
        hi.push(buffer_hi);
    }
    rayon::join(
        || par_row_outputs_recurse(first_row, mid, row_chunk, lo, widths, work),
        || par_row_outputs_recurse(first_row + mid, rows - mid, row_chunk, hi, widths, work),
    );
}

/// Apply an optimizer step in parallel over a parameter, its gradient, and any
/// number of same-length state buffers.
///
/// Every optimizer in the engine had written this by hand: a `param.len() <
/// PAR_THRESHOLD` check around a `par_chunks_mut(PAR_CHUNK)` zipped with the
/// gradient and one to three state buffers, once per dtype. Six copies of the
/// same fan-out, differing only in how many `.zip`s deep the tuple went — and
/// each of those nested `Zip` producers is its own rayon instantiation.
///
/// One element per "row", so this is [`par_row_outputs`] with every width 1,
/// plus the gradient window the caller would otherwise have to slice itself.
pub(crate) fn par_param_update<T: Send + Sync>(
    param: &mut [T],
    grad: &[T],
    state: &mut [&mut [T]],
    chunk: usize,
    work: UpdateWork<T>,
) {
    debug_assert_eq!(param.len(), grad.len());
    debug_assert!(state.iter().all(|buffer| buffer.len() == param.len()));
    let rows = param.len();
    let mut buffers: StateChunks<T> = SmallVec::with_capacity(state.len() + 1);
    buffers.push(param);
    buffers.extend(state.iter_mut().map(|buffer| &mut **buffer));
    let widths: SmallVec<[usize; 4]> = smallvec![1; buffers.len()];
    par_row_outputs(rows, chunk, &mut buffers, &widths, &|first, buffers| {
        let (param, state) = buffers
            .split_first_mut()
            .expect("the parameter is always the first buffer");
        work(param, &grad[first..first + param.len()], state);
    });
}

/// Fold `data` in parallel chunk by chunk, then combine the per-chunk results.
///
/// The erased form of `data.par_chunks(n).map(fold).reduce(|| id, combine)`.
/// Both closures are charged once per chunk, so the fold body itself — the part
/// that actually touches every element — stays a concrete type and inlines.
///
/// `fold` is handed where its chunk starts, the same way [`par_out_chunks`]
/// does, so a reduction whose answer is a *position* rather than a value can
/// use this too — `argmax` folds `(index, value)` and needs to name the index
/// it found.
///
/// `combine` must be associative and commutative for the result to be
/// independent of the split; rayon does not promise a grouping. Exact
/// operations (min, max, boolean and, bitwise or) qualify. Float addition does
/// not: `deterministic_par_sum` exists for that case.
pub(crate) fn par_fold_chunks<T, A>(
    data: &[T],
    chunk: usize,
    identity: A,
    fold: &(dyn Fn(usize, &[T]) -> A + Sync),
    combine: &(dyn Fn(A, A) -> A + Sync),
) -> A
where
    T: Sync,
    A: Copy + Send + Sync,
{
    let chunk = chunk.max(1);
    data.par_chunks(chunk)
        .enumerate()
        .map(|(nth, block)| fold(nth * chunk, block))
        .reduce(|| identity, combine)
}

/// True when `test` holds for at least one chunk of `data`.
///
/// The chunked form of `data.par_iter().any(..)`, which hands rayon one work
/// item per element to evaluate a predicate it cannot inline. Short-circuits at
/// chunk granularity, so a hit in the first chunk still stops the scan early.
pub(crate) fn par_any_chunk<T: Sync>(
    data: &[T],
    chunk: usize,
    test: &(dyn Fn(&[T]) -> bool + Sync),
) -> bool {
    data.par_chunks(chunk.max(1)).any(test)
}

/// True when `test` holds for every chunk of `data`. The counterpart of
/// [`par_any_chunk`]; spelled out rather than written as a double negation at
/// each call site.
pub(crate) fn par_all_chunk<T: Sync>(
    data: &[T],
    chunk: usize,
    test: &(dyn Fn(&[T]) -> bool + Sync),
) -> bool {
    data.par_chunks(chunk.max(1)).all(test)
}

/// True when `test` holds for every matching pair of chunks.
///
/// Short-circuits, but at chunk granularity rather than per element: rayon's
/// `all` over a zipped `par_iter` hands out one item per element and evaluates
/// the predicate behind an opaque closure, so nothing vectorizes. Here the
/// predicate sees a whole chunk and inlines.
pub(crate) fn par_all_chunks<T>(
    a: &[T],
    b: &[T],
    chunk: usize,
    test: &(dyn Fn(&[T], &[T]) -> bool + Sync),
) -> bool
where
    T: Sync,
{
    debug_assert_eq!(a.len(), b.len());
    a.par_chunks(chunk.max(1))
        .zip(b.par_chunks(chunk.max(1)))
        .all(|(a_chunk, b_chunk)| test(a_chunk, b_chunk))
}

/// Run `work(0..count)` in parallel for its effects.
///
/// The erased form of `(0..count).into_par_iter().for_each(..)`, for kernels
/// that carve their own bands out of raw pointers rather than out of slices --
/// the blocked GEMM paths, which cannot hand rayon a `&mut [T]` because the
/// bands interleave by column.
#[cfg(not(feature = "blas"))]
pub(crate) fn par_for_indexed(count: usize, work: &(dyn Fn(usize) + Sync)) {
    (0..count).into_par_iter().for_each(work);
}

/// [`par_out_chunks`] for work that can fail, stopping at the first error.
///
/// Which error is reported when several chunks fail is unspecified, exactly as
/// it was for the `map(..).collect::<Result<()>>()` this replaces.
pub(crate) fn try_par_out_chunks<T: Send, E: Send>(
    out: &mut [T],
    chunk: usize,
    work: TryOutWork<T, E>,
) -> Result<(), E> {
    if out.is_empty() {
        return Ok(());
    }
    if out.len() <= chunk || chunk == 0 {
        return work(0, out);
    }
    out.par_chunks_mut(chunk)
        .enumerate()
        .map(|(index, out_chunk)| work(index * chunk, out_chunk))
        .collect()
}

/// Run `work(0..count)` in parallel and collect the results in index order.
///
/// The erased counterpart of `(0..count).into_par_iter().map(..).collect()`,
/// for kernels that build one partial buffer per band. `count` is small — a
/// band count, not an element count — so the indirect call is charged once per
/// task and the buffer each call fills is thousands of elements.
pub(crate) fn par_map_indexed<T: Send>(count: usize, work: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    (0..count).into_par_iter().map(work).collect()
}

/// [`par_out_chunks`] over two outputs partitioned in step. Both slices must be
/// the same length and are cut at the same offsets.
pub(crate) fn par_out_chunks2<T: Send, U: Send>(
    values: &mut [T],
    indices: &mut [U],
    chunk: usize,
    work: OutWork2<T, U>,
) {
    debug_assert_eq!(values.len(), indices.len());
    if values.is_empty() {
        return;
    }
    if values.len() <= chunk || chunk == 0 {
        work(0, values, indices);
        return;
    }
    values
        .par_chunks_mut(chunk)
        .zip(indices.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(index, (value_chunk, index_chunk))| {
            work(index * chunk, value_chunk, index_chunk)
        });
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
                par_zip_chunks(input, spare, PAR_CHUNK, &|ic, oc| map_into(ic, oc, &op));
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
                par_zip_chunks(input, spare, PAR_CHUNK, &|ic, oc| op(ic, oc));
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
                par_zip_chunks2(lhs, rhs, spare, PAR_CHUNK, &|lc, rc, oc| {
                    zip_into(lc, rc, oc, &op)
                });
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
/// `chunk` is the block length, which the callers here do not agree on: a
/// gradient kernel costing nanoseconds per element wants the short
/// [`PAR_CHUNK`] blocks that keep every core fed, while the arithmetic kernels
/// cost fractions of one and want [`SIMD_PAR_CHUNK`] so the per-block overhead
/// is not most of the work.
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
    chunk: usize,
    op: F,
) -> Vec<U>
where
    A: Copy + Sync,
    B: Copy + Sync,
    U: Copy + Send + Sync,
    F: Fn(&[A], &[B], &mut [MaybeUninit<U>]) + Send + Sync,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert!(chunk > 0);
    let len = lhs.len();
    // SAFETY: forwarded to the caller by this function's own contract.
    unsafe {
        build_vec_with::<U, std::convert::Infallible, _>(len, |spare| {
            if len < threshold {
                op(lhs, rhs, spare);
            } else {
                par_zip_chunks2(lhs, rhs, spare, chunk, &|lc, rc, oc| op(lc, rc, oc));
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
                par_zip_chunks3(a, b, c, spare, PAR_CHUNK, &|ac, bc, cc, oc| {
                    zip3_into(ac, bc, cc, oc, &op)
                });
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
    fn par_out_chunks_partitions_the_output_exactly_once() {
        for len in [0usize, 1, 7, 64, 1000, 4096, 100_000] {
            for chunk in [1usize, 3, 64, 4096] {
                let mut out = vec![usize::MAX; len];
                par_out_chunks(&mut out, chunk, &|start, c| {
                    for (i, slot) in c.iter_mut().enumerate() {
                        *slot = start + i;
                    }
                });
                let expected: Vec<usize> = (0..len).collect();
                assert_eq!(out, expected, "len={len} chunk={chunk}");
            }
        }
    }

    /// `par_chunks_mut` yields nothing for an empty slice, and the kernels rely
    /// on it: several index their chunk's first row unconditionally, so calling
    /// them once with an empty chunk panics. A zero-sized `cumsum` did exactly
    /// that.
    #[test]
    fn par_out_chunks_never_runs_on_an_empty_output() {
        let mut empty: Vec<f32> = Vec::new();
        par_out_chunks(&mut empty, 8, &|_, _| panic!("must not be called"));
        par_out_chunks(&mut empty, 0, &|_, _| panic!("must not be called"));

        let (mut v, mut i): (Vec<f32>, Vec<i64>) = (Vec::new(), Vec::new());
        par_out_chunks2(&mut v, &mut i, 8, &|_, _, _| panic!("must not be called"));
        par_out_chunks2(&mut v, &mut i, 0, &|_, _, _| panic!("must not be called"));
    }

    #[test]
    fn par_out_chunks2_cuts_both_outputs_at_the_same_offsets() {
        for len in [0usize, 1, 5, 4096, 20_000] {
            let (mut values, mut indices) = (vec![0u32; len], vec![0i64; len]);
            par_out_chunks2(&mut values, &mut indices, 64, &|start, v, i| {
                assert_eq!(v.len(), i.len());
                for (offset, (value, index)) in v.iter_mut().zip(i.iter_mut()).enumerate() {
                    *value = (start + offset) as u32;
                    *index = (start + offset) as i64;
                }
            });
            assert_eq!(values, (0..len as u32).collect::<Vec<_>>(), "{len}");
            assert_eq!(indices, (0..len as i64).collect::<Vec<_>>(), "{len}");
        }
    }

    #[test]
    fn par_fold_chunks_and_par_map_indexed_match_their_sequential_forms() {
        for len in [0usize, 1, 1000, 50_000] {
            let data: Vec<i64> = (0..len as i64).collect();
            let total = par_fold_chunks(&data, 128, 0i64, &|_, c| c.iter().sum(), &|a, b| a + b);
            assert_eq!(total, data.iter().sum::<i64>(), "{len}");
        }
        assert_eq!(par_map_indexed(0, &|i: usize| i), Vec::<usize>::new());
        assert_eq!(
            par_map_indexed(37, &|i: usize| i * i),
            (0..37).map(|i| i * i).collect::<Vec<_>>()
        );
    }

    /// The split must be invisible: every element sees the same parameter,
    /// gradient and state values it would have seen running sequentially, and
    /// the state buffers stay aligned with the parameter they belong to.
    #[test]
    fn par_param_update_matches_a_sequential_step_for_every_arity() {
        for len in [0usize, 1, 1023, 1024, 1025, 70_000] {
            for arity in 0..=3usize {
                let param: Vec<f64> = (0..len).map(|i| i as f64 * 0.5).collect();
                let grad: Vec<f64> = (0..len).map(|i| 1.0 / (i as f64 + 1.0)).collect();
                let states: Vec<Vec<f64>> = (0..arity)
                    .map(|s| (0..len).map(|i| (s * len + i) as f64).collect())
                    .collect();

                // One step, written once, applied both ways.
                let step = |p: &mut [f64], g: &[f64], state: &mut [&mut [f64]]| {
                    for i in 0..p.len() {
                        let mut delta = g[i];
                        for buffer in state.iter_mut() {
                            buffer[i] = 0.9 * buffer[i] + g[i];
                            delta += buffer[i];
                        }
                        p[i] -= 0.01 * delta;
                    }
                };

                let (mut seq_param, mut seq_states) = (param.clone(), states.clone());
                let mut seq_refs: Vec<&mut [f64]> =
                    seq_states.iter_mut().map(|s| s.as_mut_slice()).collect();
                step(&mut seq_param, &grad, &mut seq_refs);

                let (mut par_param, mut par_states) = (param.clone(), states.clone());
                let mut par_refs: Vec<&mut [f64]> =
                    par_states.iter_mut().map(|s| s.as_mut_slice()).collect();
                par_param_update(&mut par_param, &grad, &mut par_refs, 1024, &step);

                assert_eq!(par_param, seq_param, "param len={len} arity={arity}");
                assert_eq!(par_states, seq_states, "state len={len} arity={arity}");
            }
        }
    }

    /// Each leaf must be a whole chunk, so a kernel that reasons about its
    /// window's alignment is not handed a ragged one in the middle.
    #[test]
    fn par_param_update_splits_only_on_chunk_boundaries() {
        use std::sync::Mutex;
        let len = 70_000;
        let chunk = 1024;
        let (mut param, grad) = (vec![0.0f32; len], vec![0.0f32; len]);
        let windows = Mutex::new(Vec::new());
        par_param_update(&mut param, &grad, &mut [], chunk, &|p, _, _| {
            windows.lock().expect("lock").push(p.len());
        });
        let mut widths = windows.into_inner().expect("lock");
        widths.sort_unstable();
        assert_eq!(widths.iter().sum::<usize>(), len);
        // Every window is a full chunk but the last, which holds the remainder.
        let tail = len % chunk;
        assert_eq!(widths[0], tail, "{widths:?}");
        assert!(widths[1..].iter().all(|&w| w == chunk), "{widths:?}");
    }

    #[test]
    fn outputs_per_task_scales_with_the_cost_of_one_output() {
        assert!(outputs_per_task(1) > outputs_per_task(1024));
        // A single output can cost more than a whole task's budget; the floor
        // keeps the chunk width usable rather than zero.
        assert_eq!(outputs_per_task(usize::MAX), 1);
        assert_eq!(outputs_per_task(0), outputs_per_task(1));
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
