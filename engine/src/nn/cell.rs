// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! What the fused recurrent cells share.
//!
//! [`crate::nn::lstm_cell`] and [`crate::nn::gru_cell`] are the same shape of
//! kernel over different arithmetic: a `(batch, gates * hidden)` buffer
//! activated in blocks, then a few vectorizable passes. The partitioning rule
//! and the float64 fallbacks are the parts that would otherwise be written
//! twice, so they are written here once.

use crate::ops::map::SIMD_PAR_CHUNK;
use crate::ops::util::stable_sigmoid_f64;
use std::mem::MaybeUninit;

/// The chunk width to hand [`crate::ops::map::par_out_chunks`] for `total`
/// elements laid out in rows of `row_width`.
///
/// Below `threshold` the whole buffer is one chunk, which `par_out_chunks`
/// runs inline without reaching rayon at all. That is the case that matters:
/// a recurrent step's tensors are small by construction, and forking a pool
/// once per timestep would cost more than the arithmetic it splits. The
/// thresholds are the ones the composed operations used, so the point at which
/// this starts using every core is the point at which they did.
///
/// Above it the bands are whole rows. Which gate an element belongs to is
/// decided by where it sits within its row, so a split anywhere else would
/// hand a task half of one gate and half of the next -- which would not fail
/// loudly. It would activate the wrong elements with the wrong function.
#[inline]
pub(crate) fn band_width(total: usize, row_width: usize, threshold: usize) -> usize {
    if total < threshold || row_width == 0 {
        return total;
    }
    row_width * (SIMD_PAR_CHUNK / row_width).max(1)
}

/// `sigmoid` over a block of float64, element at a time.
///
/// There is no vectorized float64 transcendental here or in the elementwise
/// kernels, so the composed form these cells replace was scalar too. Same
/// function, same rounding.
#[inline]
pub(crate) fn sigmoid_block_f64(src: &[f64], dst: &mut [MaybeUninit<f64>]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        out.write(stable_sigmoid_f64(x));
    }
}

/// `tanh` over a block of float64. See [`sigmoid_block_f64`].
#[inline]
pub(crate) fn tanh_block_f64(src: &[f64], dst: &mut [MaybeUninit<f64>]) {
    for (out, &x) in dst.iter_mut().zip(src) {
        out.write(x.tanh());
    }
}
