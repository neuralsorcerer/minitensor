// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float32 convolutions whose every entry is the float32 nearest its exact
//! value.
//!
//! A convolution, and each of its gradients, is a set of sums of products of
//! two float32s, each product exact in float64. Added in float64 in any
//! order, a sum of `n` of them is off the exact one by at most
//! `γ_n = n·u / (1 - n·u)` of the sum of their magnitudes, `u = 2^-53`; where
//! both ends of that bound narrow to one float32, rounding being monotonic,
//! so does the exact sum, and where they do not -- one entry in some
//! thousands of ordinary data, more where an entry cancels far below its
//! terms -- the entry is the exact sum of its own products, gathered from the
//! operands.
//!
//! * The output is lowered to columns and multiplied as a matrix product
//!   rounded once, the bias its addend. A group with few output channels is
//!   added up straight from the input, each plane against the kernel's
//!   magnitudes times the largest value it reads, an entry that leaves open
//!   against its own terms' magnitudes.
//! * The input gradient -- a transposed convolution's output, its bias added
//!   in -- is the float64 product of the kernel and the incoming gradient
//!   scattered back onto the grid, each value carrying the products of the
//!   2-norms of the kernel's column and the gradient's position it took, a
//!   bound on the magnitudes of their terms; or, for a group with few output
//!   channels, the scatter straight from the signal, bounded as the direct
//!   output is.
//! * The weight gradient is a sum over every position of every image. It is
//!   added in float64 a short run at a time, the runs carried beside one
//!   another exactly by two-sums, so its bound is a run's, against the
//!   product of the 2-norms of the two channels it pairs.
//!
//! Where an entry's bound reaches a halfway point -- an exact tie, which a
//! sum of a few short values often is -- and its terms' magnitudes stay
//! under `2^53` of the product of the operands' grains, the float64 sum took
//! no rounding and is the exact one, which is rounded from there.

use super::{
    COL_BLOCK_BYTES, ConvGeometry, direct_pays, images_per_block, in_bounds_range, lower_positions,
};
use crate::error::Result;
use crate::ops::arithmetic::exact_sum::{dot_f32, exact_dot_to};
use crate::ops::arithmetic::rounded_once::two_sum;
use crate::ops::linalg::{Affine, grain_of, rounded_affine_products_in_engine_f32, settles};
use crate::ops::map::{par_map_indexed, par_out_chunks, par_out_chunks2};
use crate::ops::provider::Storage;

/// Float64's unit roundoff, `2^-53`.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// Positions a run of the weight gradient adds up plainly before it is
/// carried beside the others.
const RUN: usize = 256;

/// Lanes a direct weight gradient adds a run in, so the run vectorizes.
const LANES: usize = 8;

/// Terms a lane of the direct weight gradient takes before its run is
/// carried, at least: few, so its bound is that of a short sum.
const DEPTH: usize = 16;

/// Rows of the lowered columns a task of the lowered weight gradient takes.
const BAND: usize = 64;

/// `2^53`, short by `2^-18` of itself for the roundings of the bound held
/// against it: a sum of whole multiples of a grain whose magnitudes stay
/// under this many of it is a whole multiple short of `2^53` of them at
/// every step, so a float64, and took no rounding.
const WHOLE: f64 = ((1u64 << 53) - (1u64 << 35)) as f64;

/// `γ_n` for a sum of at most `terms` exact products, with room for the
/// roundings of the bound it scales: of the magnitudes or norms it is taken
/// against, under `(1 + 4n·u)` of them, and of its own working, `2^-40`.
fn scale(terms: usize) -> f64 {
    let reach = terms as f64 * UNIT;
    reach / (1.0 - reach) * (1.0 + 4.0 * reach + 2f64.powi(-40))
}

/// `sum + a · b` for a product `a · b` exact in float64, fused where `FMA`:
/// the one rounding either way.
#[inline(always)]
fn add_exact<const FMA: bool>(sum: f64, a: f64, b: f64) -> f64 {
    if FMA { a.mul_add(b, sum) } else { sum + a * b }
}

/// `values` widened into `wide`; the largest of their magnitudes, a NaN
/// passed over.
#[inline(always)]
fn widen_into(wide: &mut [f64], values: &[f32]) -> f64 {
    let mut largest = 0.0f64;
    for (slot, &value) in wide.iter_mut().zip(values) {
        let value = f64::from(value);
        *slot = value;
        largest = largest.max(value.abs());
    }
    largest
}

/// The room a product of two 2-norms needs for their own roundings: each
/// the root of a float64 sum of exact squares, `count` of them between the
/// two.
fn norms_margin(count: usize) -> f64 {
    1.0 + 2.0 * (count + 8) as f64 * UNIT
}

/// The convolution of `input` by `weight`, `bias` added, every entry the
/// float32 nearest its exact value; the shapes are those
/// [`super::convolve_plain`] takes.
pub(super) fn convolve(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    geometry: &ConvGeometry,
) -> Result<Vec<f32>> {
    let ConvGeometry {
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        groups,
        ..
    } = *geometry;
    let ohw = output_height * output_width;
    let mut output = vec![0.0f32; batch_size * out_channels * ohw];
    if output.is_empty() {
        return Ok(output);
    }
    if direct_pays(geometry) {
        par_out_chunks(&mut output, ohw, &|first, plane| {
            let index = first / ohw;
            convolve_plane(
                plane,
                (index / out_channels, index % out_channels),
                input,
                weight,
                bias,
                geometry,
            );
        });
        return Ok(output);
    }

    // Blocked by output position as the float64 path is, for the reasons it
    // gives; each group's block is a product rounded once, its bias the
    // product's addend.
    let kh_kw = kernel_h * kernel_w;
    let k_dim = in_channels * kh_kw;
    let group_out = out_channels / groups;
    let group_k = (in_channels / groups) * kh_kw;
    let per_column = (k_dim * std::mem::size_of::<f32>()).max(1);
    let positions_total = batch_size * ohw;
    let budget = (COL_BLOCK_BYTES / per_column).clamp(1, positions_total);
    let block_n = if budget >= output_width {
        (budget / output_width) * output_width
    } else {
        budget
    };
    let mut cols = vec![0.0f32; k_dim * block_n];
    let mut products = vec![0.0f32; out_channels * block_n];

    let mut first = 0;
    while first < positions_total {
        let last = (first + block_n).min(positions_total);
        let width = last - first;
        lower_positions::<f32>(input, &mut cols, first, last, geometry);
        for g in 0..groups {
            rounded_affine_products_in_engine_f32(
                1,
                group_out,
                group_k,
                width,
                (
                    &weight[g * group_out * group_k..][..group_out * group_k],
                    Storage::RowMajor,
                ),
                (
                    &cols[g * group_k * width..][..group_k * width],
                    Storage::RowMajor,
                ),
                bias.map(|bias| Affine {
                    alpha: 1.0,
                    beta: 1.0,
                    values: &bias[g * group_out..][..group_out],
                    strides: [0, 1, 0],
                }),
                &mut products[g * group_out * width..][..group_out * width],
            )?;
        }

        // `[C_out, width]` into `[N, C_out, oh, ow]`, a plane per task as the
        // float64 path places its block.
        let first_image = first / ohw;
        let last_image = (last - 1) / ohw + 1;
        let region = &mut output[first_image * out_channels * ohw..last_image * out_channels * ohw];
        let products = &products;
        par_out_chunks(region, ohw, &|start, plane| {
            let index = start / ohw;
            let (n, oc) = (first_image + index / out_channels, index % out_channels);
            let image_start = n * ohw;
            let from = first.max(image_start) - image_start;
            let to = last.min(image_start + ohw) - image_start;
            plane[from..to].copy_from_slice(
                &products[oc * width + (image_start + from - first)..][..to - from],
            );
        });
        first = last;
    }
    Ok(output)
}

/// Output plane `(n, oc)` of [`convolve`] straight from the input: each
/// output's taps added in float64, then settled against the plane's bound.
fn convolve_plane(
    plane: &mut [f32],
    (n, oc): (usize, usize),
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    geometry: &ConvGeometry,
) {
    let ConvGeometry {
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        groups,
        ..
    } = *geometry;
    let mut sums = vec![0.0f64; plane.len()];
    let largest = gather_sums(&mut sums, (n, oc), input, weight, geometry);
    let group_in = in_channels / groups;
    let terms = group_in * kernel_h * kernel_w;
    let taps = &weight[oc * terms..][..terms];
    // Each output's terms are at most the kernel's magnitudes times the
    // largest value they read.
    let bound = taps.iter().map(|&tap| f64::from(tap).abs()).sum::<f64>() * largest;
    let grains = || {
        let plane = input_height * input_width;
        let group = oc / (out_channels / groups);
        let reads = &input[(n * in_channels + group * group_in) * plane..][..group_in * plane];
        WHOLE * grain_of(taps.iter().copied()) * grain_of(reads.iter().copied())
    };
    settle_plane(
        plane,
        &sums,
        (
            scale(terms + 1),
            bias.map_or(0.0, |bias| f64::from(bias[oc])),
        ),
        |_| bound,
        grains,
        |position| forward_terms((n, oc), position, input, weight, geometry),
    );
}

/// Each entry of `plane` from its float64 `sum` plus `addend`, the sum
/// within `scale` times `bound(position)` of the exact one, the bound at
/// least the sum of its terms' magnitudes: the float32 nearest that where
/// both ends of the bound narrow alike.
///
/// The rest are found again from their own `terms`: against the sum of their
/// magnitudes, which is tighter; then from their sum itself where that is
/// under `grains()` -- [`WHOLE`] times the grains of the two operands the
/// terms multiply, every product a whole multiple of theirs, so the sum took
/// no rounding; and otherwise as the exact sum of the products and the
/// addend. A sum of products that cancel, or of zeros, is a positive zero.
fn settle_plane(
    plane: &mut [f32],
    sums: &[f64],
    (scale, addend): (f64, f64),
    bound: impl Fn(usize) -> f64,
    grains: impl Fn() -> f64,
    terms: impl Fn(usize) -> (Vec<f32>, Vec<f32>),
) {
    // The addition of the addend rounds once more, by at most `2^-53` of its
    // result, which the margin `settles` keeps for its own roundings covers.
    let mut open = false;
    for (position, (slot, &sum)) in plane.iter_mut().zip(sums).enumerate() {
        let value = sum + addend;
        *slot = value as f32;
        open |= !settles(value, scale * bound(position));
    }
    if !open {
        return;
    }
    // A sum that took no rounding is rounded once with the addend.
    let exact_sum = |sum: f64| {
        if addend == 0.0 {
            (sum + 0.0) as f32
        } else {
            exact_dot_to::<f32>([(1.0, sum), (1.0, addend)].into_iter())
        }
    };
    let mut whole = None;
    for (position, (slot, &sum)) in plane.iter_mut().zip(sums).enumerate() {
        let bound = bound(position);
        if settles(sum + addend, scale * bound) {
            continue;
        }
        let whole = *whole.get_or_insert_with(&grains);
        if bound < whole {
            *slot = exact_sum(sum);
            continue;
        }
        let (mut taps, mut reads) = terms(position);
        let magnitude: f64 = taps
            .iter()
            .zip(&reads)
            .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
            .sum();
        if settles(sum + addend, scale * magnitude) {
            continue;
        }
        *slot = if magnitude < whole {
            exact_sum(sum)
        } else {
            // The addend is a float32, a last product with one.
            taps.push(addend as f32);
            reads.push(1.0);
            dot_f32(&taps, &reads)
        };
    }
}

/// The taps of output plane `(n, oc)` added into `sums`; the largest
/// magnitude among the values they read.
fn gather_sums(
    sums: &mut [f64],
    plane: (usize, usize),
    input: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { gather_sums_avx2(sums, plane, input, weight, geometry) };
    }
    gather_sums_body::<false>(sums, plane, input, weight, geometry)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn gather_sums_avx2(
    sums: &mut [f64],
    plane: (usize, usize),
    input: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    gather_sums_body::<true>(sums, plane, input, weight, geometry)
}

#[inline(always)]
fn gather_sums_body<const FMA: bool>(
    sums: &mut [f64],
    (n, oc): (usize, usize),
    input: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    let ConvGeometry {
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        groups,
        ..
    } = *geometry;
    let group_in = in_channels / groups;
    let group = oc / (out_channels / groups);
    let input_plane = input_height * input_width;
    // Each plane widened once for all of its taps.
    let mut wide = vec![0.0f64; input_plane];
    let mut largest = 0.0f64;
    for local in 0..group_in {
        let ic = group * group_in + local;
        let source = &input[(n * in_channels + ic) * input_plane..][..input_plane];
        largest = largest.max(widen_into(&mut wide, source));
        for ky in 0..kernel_h {
            let ky_off = ky * dilation.0;
            let (oh_lo, oh_hi) =
                in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
            for kx in 0..kernel_w {
                let kx_off = kx * dilation.1;
                let (ow_lo, ow_hi) =
                    in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
                if oh_lo >= oh_hi || ow_lo >= ow_hi {
                    continue;
                }
                let tap =
                    f64::from(weight[((oc * group_in + local) * kernel_h + ky) * kernel_w + kx]);
                let span = ow_hi - ow_lo;
                for oh in oh_lo..oh_hi {
                    let row =
                        &wide[(oh * stride.0 + ky_off - padding.0) * input_width..][..input_width];
                    let at = oh * output_width + ow_lo;
                    let out = &mut sums[at..at + span];
                    if stride.1 == 1 {
                        let start = ow_lo + kx_off - padding.1;
                        for (sum, &x) in out.iter_mut().zip(&row[start..start + span]) {
                            *sum = add_exact::<FMA>(*sum, tap, x);
                        }
                    } else {
                        for (i, sum) in out.iter_mut().enumerate() {
                            let x = row[(ow_lo + i) * stride.1 + kx_off - padding.1];
                            *sum = add_exact::<FMA>(*sum, tap, x);
                        }
                    }
                }
            }
        }
    }
    largest
}

/// Where tap `k` of an axis reading position `i` of the grid sits in the
/// signal: `o` with `o · stride + k · dilation - padding = i`, if there is
/// one in `[0, extent)`.
#[inline]
fn signal_index(
    i: usize,
    k: usize,
    (stride, padding, dilation): (usize, usize, usize),
    extent: usize,
) -> Option<usize> {
    let shifted = (i + padding).checked_sub(k * dilation)?;
    (shifted.is_multiple_of(stride) && shifted / stride < extent).then_some(shifted / stride)
}

/// Where position `o` of the signal reads with tap `k` on the grid, if that
/// is inside `[0, extent)`.
#[inline]
fn grid_index(
    o: usize,
    k: usize,
    (stride, padding, dilation): (usize, usize, usize),
    extent: usize,
) -> Option<usize> {
    (o * stride + k * dilation)
        .checked_sub(padding)
        .filter(|&i| i < extent)
}

/// The terms of output `position` of plane `(n, oc)`: each tap of the kernel
/// beside the value it reads.
fn forward_terms(
    (n, oc): (usize, usize),
    position: usize,
    input: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> (Vec<f32>, Vec<f32>) {
    let ConvGeometry {
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_width,
        stride,
        padding,
        dilation,
        groups,
        ..
    } = *geometry;
    let group_in = in_channels / groups;
    let group = oc / (out_channels / groups);
    let (oh, ow) = (position / output_width, position % output_width);
    let (mut taps, mut reads) = (Vec::new(), Vec::new());
    for local in 0..group_in {
        let ic = group * group_in + local;
        for ky in 0..kernel_h {
            let rows = (stride.0, padding.0, dilation.0);
            let Some(ih) = grid_index(oh, ky, rows, input_height) else {
                continue;
            };
            for kx in 0..kernel_w {
                let columns = (stride.1, padding.1, dilation.1);
                let Some(iw) = grid_index(ow, kx, columns, input_width) else {
                    continue;
                };
                taps.push(weight[((oc * group_in + local) * kernel_h + ky) * kernel_w + kx]);
                reads.push(input[((n * in_channels + ic) * input_height + ih) * input_width + iw]);
            }
        }
    }
    (taps, reads)
}

/// `col2im(weight^T @ source)`, `bias` added to each channel of the grid,
/// every entry the float32 nearest its exact value; the shapes are those
/// [`super::scatter_columns`] takes.
pub(super) fn scatter(
    source: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    geometry: &ConvGeometry,
) -> Vec<f32> {
    let ConvGeometry {
        batch_size: batch,
        in_channels,
        input_height: in_h,
        input_width: in_w,
        out_channels,
        kernel_h,
        kernel_w,
        output_height: out_h,
        output_width: out_w,
        groups,
        ..
    } = *geometry;
    let plane = in_h * in_w;
    let mut destination = vec![0.0f32; batch * in_channels * plane];
    if destination.is_empty() {
        return destination;
    }
    let ohw = out_h * out_w;
    let kh_kw = kernel_h * kernel_w;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let group_k = group_in * kh_kw;
    let addend = |channel: usize| bias.map_or(0.0, |bias| f64::from(bias[channel]));
    // The grains of the kernel's taps grid channel `ic` reads, and of the
    // signal's planes it reads them against in image `n`.
    let grains = |n: usize, ic: usize| {
        move || {
            let (group, local) = (ic / group_in, ic % group_in);
            let taps = (group * group_out..(group + 1) * group_out).flat_map(|oc| {
                weight[(oc * group_in + local) * kh_kw..][..kh_kw]
                    .iter()
                    .copied()
            });
            let reads = &source[(n * out_channels + group * group_out) * ohw..][..group_out * ohw];
            WHOLE * grain_of(taps) * grain_of(reads.iter().copied())
        }
    };
    let terms_of = |n: usize, ic: usize| {
        move |position: usize| scatter_terms((n, ic), position, source, weight, geometry)
    };

    if ohw == 0 || direct_pays(geometry) {
        par_out_chunks(&mut destination, plane, &|first, target| {
            let index = first / plane;
            let (n, ic) = (index / in_channels, index % in_channels);
            let mut sums = vec![0.0f64; plane];
            let largest = if ohw > 0 {
                spread_sums(&mut sums, (n, ic), source, weight, geometry)
            } else {
                0.0
            };
            // Each entry's terms are at most the magnitudes of the taps the
            // plane takes times the largest value of the signal they read.
            let (group, local) = (ic / group_in, ic % group_in);
            let taps: f64 = (group * group_out..(group + 1) * group_out)
                .flat_map(|oc| weight[(oc * group_in + local) * kh_kw..][..kh_kw].iter())
                .map(|&tap| f64::from(tap).abs())
                .sum();
            let bound = taps * largest;
            // One sum of every tap of every output channel of the group.
            settle_plane(
                target,
                &sums,
                (scale(group_out * kh_kw + 1), addend(ic)),
                |_| bound,
                grains(n, ic),
                terms_of(n, ic),
            );
        });
        return destination;
    }

    // The kernel widened with each of its columns made a row, `[groups,
    // group_k, group_out]`, so a run of a group's grid channels reads a block
    // of rows; and the 2-norm of each.
    let mut columns_of_kernel = vec![0.0f64; groups * group_k * group_out];
    for (index, slot) in columns_of_kernel.iter_mut().enumerate() {
        let (row, o) = (index / group_out, index % group_out);
        let (g, k) = (row / group_k, row % group_k);
        *slot = f64::from(weight[(g * group_out + o) * group_k + k]);
    }
    let tap_norms: Vec<f64> = columns_of_kernel
        .chunks_exact(group_out)
        .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt())
        .collect();
    // The signal widened, and the 2-norm of each of its positions across each
    // group's channels, `[N, groups, OH * OW]`.
    let wide_source = crate::ops::map::unary_map(source, f64::from);
    let mut position_norms = vec![0.0f64; batch * groups * ohw];
    par_out_chunks(&mut position_norms, ohw, &|start, part| {
        let (n, g) = (start / ohw / groups, start / ohw % groups);
        for o in 0..group_out {
            let line = &wide_source[(n * out_channels + g * group_out + o) * ohw..][..ohw];
            for (slot, &value) in part.iter_mut().zip(line) {
                *slot += value * value;
            }
        }
        for slot in part.iter_mut() {
            *slot = slot.sqrt();
        }
    });
    // An entry is a sum of up to `kh_kw` of the product's values, each a sum
    // over the group's output channels: within `γ` of that many roundings
    // between them of the magnitudes of its terms, which for each tap are at
    // most the product of its norm and its position's.
    let bound_scale = scale(group_out + kh_kw + 2) * norms_margin(2 * group_out + kh_kw);

    // A task takes a run of grid planes, a few per thread across the batch,
    // its columns no more than some hundreds of kilobytes: for each stretch
    // of them in one image and group, the product of their kernel rows with
    // the image's signal, scattered back onto the planes beside the bounds
    // their terms carry, each plane then settled.
    let threads = crate::parallel::current_num_threads().max(1);
    let fits = (1usize << 15) / (kh_kw * ohw).max(1);
    let per_task = (batch * in_channels)
        .div_ceil(4 * threads)
        .clamp(1, fits.max(1));
    par_out_chunks(&mut destination, per_task * plane, &|start, target| {
        let first_plane = start / plane;
        let mut columns = Vec::new();
        let mut done = 0;
        while done < target.len() / plane {
            let index = first_plane + done;
            let (n, ic) = (index / in_channels, index % in_channels);
            let (g, local) = (ic / group_in, ic % group_in);
            // The planes from here on in the same image and group.
            let stretch = (target.len() / plane - done).min(group_in - local);
            let rows = stretch * kh_kw;
            columns.clear();
            columns.resize(rows * ohw, 0.0f64);
            // SAFETY: the kernel's rows `[g*group_k + local*kh_kw, ..)` are
            // `[rows, group_out]`, the signal's planes of image `n` from
            // `g*group_out` are `[group_out, ohw]`, and `columns` is `[rows,
            // ohw]`, each inside its buffer.
            unsafe {
                crate::ops::linalg::gemm_f64(
                    rows,
                    group_out,
                    ohw,
                    columns_of_kernel
                        .as_ptr()
                        .add((g * group_k + local * kh_kw) * group_out),
                    wide_source
                        .as_ptr()
                        .add((n * out_channels + g * group_out) * ohw),
                    columns.as_mut_ptr(),
                );
            }
            let norms = &position_norms[(n * groups + g) * ohw..][..ohw];
            for offset in 0..stretch {
                let channel = ic + offset;
                let mut sums = vec![0.0f64; plane];
                let mut bounds = vec![0.0f64; plane];
                let taps = &columns[offset * kh_kw * ohw..][..kh_kw * ohw];
                let tap_norms = &tap_norms[(g * group_k + (local + offset) * kh_kw)..][..kh_kw];
                scatter_columns_back(&mut sums, &mut bounds, taps, tap_norms, norms, geometry);
                settle_plane(
                    &mut target[(done + offset) * plane..][..plane],
                    &sums,
                    (bound_scale, addend(channel)),
                    |position| bounds[position],
                    grains(n, channel),
                    terms_of(n, channel),
                );
            }
            done += stretch;
        }
    });
    destination
}

/// One grid plane's `columns` -- a row per tap, over the signal's positions
/// -- added back onto it: each value into `sums`, and the product of its
/// tap's norm and its position's into `bounds`.
fn scatter_columns_back(
    sums: &mut [f64],
    bounds: &mut [f64],
    columns: &[f64],
    tap_norms: &[f64],
    position_norms: &[f64],
    geometry: &ConvGeometry,
) {
    let ConvGeometry {
        input_height: in_h,
        input_width: in_w,
        kernel_w,
        output_height: out_h,
        output_width: out_w,
        stride,
        padding,
        dilation,
        ..
    } = *geometry;
    let ohw = out_h * out_w;
    for (tap, &tap_norm) in tap_norms.iter().enumerate() {
        let ky_off = (tap / kernel_w) * dilation.0;
        let kx_off = (tap % kernel_w) * dilation.1;
        let (oh_lo, oh_hi) = in_bounds_range(ky_off, padding.0, in_h, stride.0, out_h);
        let (ow_lo, ow_hi) = in_bounds_range(kx_off, padding.1, in_w, stride.1, out_w);
        if oh_lo >= oh_hi || ow_lo >= ow_hi {
            continue;
        }
        let row = &columns[tap * ohw..][..ohw];
        let span = ow_hi - ow_lo;
        for oh in oh_lo..oh_hi {
            let at = (oh * stride.0 + ky_off - padding.0) * in_w;
            let values = &row[oh * out_w + ow_lo..][..span];
            let norms = &position_norms[oh * out_w + ow_lo..][..span];
            if stride.1 == 1 {
                let start = at + ow_lo + kx_off - padding.1;
                let out = sums[start..start + span]
                    .iter_mut()
                    .zip(&mut bounds[start..start + span]);
                for ((sum, bound), (&value, &norm)) in out.zip(values.iter().zip(norms)) {
                    *sum += value;
                    *bound += tap_norm * norm;
                }
            } else {
                for (i, (&value, &norm)) in values.iter().zip(norms).enumerate() {
                    let at = at + (ow_lo + i) * stride.1 + kx_off - padding.1;
                    sums[at] += value;
                    bounds[at] += tap_norm * norm;
                }
            }
        }
    }
}

/// The terms grid plane `(n, ic)` of [`scatter`] takes, straight from the
/// signal, each added into `sums`; the largest magnitude among the values of
/// the signal they read.
fn spread_sums(
    sums: &mut [f64],
    plane: (usize, usize),
    source: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { spread_sums_avx2(sums, plane, source, weight, geometry) };
    }
    spread_sums_body::<false>(sums, plane, source, weight, geometry)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn spread_sums_avx2(
    sums: &mut [f64],
    plane: (usize, usize),
    source: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    spread_sums_body::<true>(sums, plane, source, weight, geometry)
}

#[inline(always)]
fn spread_sums_body<const FMA: bool>(
    sums: &mut [f64],
    (n, ic): (usize, usize),
    source: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> f64 {
    let ConvGeometry {
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        groups,
        ..
    } = *geometry;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let (group, local) = (ic / group_in, ic % group_in);
    let plane_out = output_height * output_width;
    // Each plane of the signal widened once for all of its taps.
    let mut signal = vec![0.0f64; plane_out];
    let mut largest = 0.0f64;
    for oc in group * group_out..(group + 1) * group_out {
        let read = &source[(n * out_channels + oc) * plane_out..][..plane_out];
        largest = largest.max(widen_into(&mut signal, read));
        for ky in 0..kernel_h {
            let ky_off = ky * dilation.0;
            let (oh_lo, oh_hi) =
                in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
            for kx in 0..kernel_w {
                let kx_off = kx * dilation.1;
                let (ow_lo, ow_hi) =
                    in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
                if oh_lo >= oh_hi || ow_lo >= ow_hi {
                    continue;
                }
                let tap =
                    f64::from(weight[((oc * group_in + local) * kernel_h + ky) * kernel_w + kx]);
                let span = ow_hi - ow_lo;
                for oh in oh_lo..oh_hi {
                    let from = &signal[oh * output_width + ow_lo..][..span];
                    let at = (oh * stride.0 + ky_off - padding.0) * input_width;
                    if stride.1 == 1 {
                        let start = at + ow_lo + kx_off - padding.1;
                        for (sum, &g) in sums[start..start + span].iter_mut().zip(from) {
                            *sum = add_exact::<FMA>(*sum, tap, g);
                        }
                    } else {
                        for (i, &g) in from.iter().enumerate() {
                            let at = at + (ow_lo + i) * stride.1 + kx_off - padding.1;
                            sums[at] = add_exact::<FMA>(sums[at], tap, g);
                        }
                    }
                }
            }
        }
    }
    largest
}

/// The terms of grid `position` of plane `(n, ic)` of [`scatter`]: each tap
/// of the kernel that reaches it beside the value of the signal it carries.
fn scatter_terms(
    (n, ic): (usize, usize),
    position: usize,
    source: &[f32],
    weight: &[f32],
    geometry: &ConvGeometry,
) -> (Vec<f32>, Vec<f32>) {
    let ConvGeometry {
        in_channels,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        groups,
        ..
    } = *geometry;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let (group, local) = (ic / group_in, ic % group_in);
    let (ih, iw) = (position / input_width, position % input_width);
    let (mut taps, mut reads) = (Vec::new(), Vec::new());
    for oc in group * group_out..(group + 1) * group_out {
        for ky in 0..kernel_h {
            let rows = (stride.0, padding.0, dilation.0);
            let Some(oh) = signal_index(ih, ky, rows, output_height) else {
                continue;
            };
            for kx in 0..kernel_w {
                let columns = (stride.1, padding.1, dilation.1);
                let Some(ow) = signal_index(iw, kx, columns, output_width) else {
                    continue;
                };
                taps.push(weight[((oc * group_in + local) * kernel_h + ky) * kernel_w + kx]);
                reads.push(
                    source[((n * out_channels + oc) * output_height + oh) * output_width + ow],
                );
            }
        }
    }
    (taps, reads)
}

/// `source @ im2col(image)^T`, every entry the float32 nearest its exact
/// value; the shapes are those [`super::column_weight_gradient`] takes.
pub(super) fn weight_gradient(image: &[f32], source: &[f32], geometry: &ConvGeometry) -> Vec<f32> {
    let ConvGeometry {
        batch_size: batch,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        groups,
        ..
    } = *geometry;
    let ohw = output_height * output_width;
    let kh_kw = kernel_h * kernel_w;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let group_k = group_in * kh_kw;
    let mut gradient = vec![0.0f32; out_channels * group_k];
    if gradient.is_empty() || ohw == 0 || batch == 0 {
        // A sum of no products: a positive zero.
        return gradient;
    }

    // The sums and errors, how many roundings a term of a run passes
    // through, and how many carries an entry takes.
    let (sums, errors, depth, carries) = if direct_pays(geometry) {
        let mut sums = vec![0.0f64; gradient.len()];
        let mut errors = vec![0.0f64; gradient.len()];
        par_out_chunks2(&mut sums, &mut errors, kh_kw, &|first, sums, errors| {
            weight_slice_sums(sums, errors, first / kh_kw, image, source, geometry);
        });
        // A lane's run, and the merge of the lanes; and a carry each time a
        // lane fills, from the share of every row its first lane takes.
        let depth = DEPTH + output_width.div_ceil(LANES) + 3;
        let shares = batch * output_height * output_width.div_ceil(LANES);
        (sums, errors, depth, shares / DEPTH + 1)
    } else {
        let (sums, errors, carries) = lowered_weight_sums(image, source, geometry);
        (sums, errors, RUN, carries)
    };

    // Each entry's terms pair a channel of the signal with one of the image,
    // each position of the signal with a different one of the image, so the
    // magnitudes of the terms are at most the product of the two channels'
    // 2-norms over every image. A run is added plainly, within `γ` of that
    // run's, and the runs carried by two-sums whose errors are added
    // plainly: within `γ_J · u · J` of the total's magnitudes for `J`
    // carries.
    let input_plane = input_height * input_width;
    let signal_norms = channel_norms(source, batch, out_channels, ohw);
    let image_norms = channel_norms(image, batch, in_channels, input_plane);
    let margin = norms_margin(batch * (ohw + input_plane));
    let carried = 2.0 * (carries as f64 * UNIT).powi(2);
    let factor = (scale(depth) + carried) * margin;
    // The grain of each channel of the two, found once an entry needs them.
    let signal_grains = std::sync::OnceLock::new();
    let image_grains = std::sync::OnceLock::new();
    let (sums, errors) = (&sums, &errors);
    par_out_chunks(&mut gradient, group_k, &|first, row| {
        let oc = first / group_k;
        for (k, slot) in row.iter_mut().enumerate() {
            let index = oc * group_k + k;
            let value = sums[index] + errors[index];
            let ic = (oc / group_out) * group_in + k / kh_kw;
            let magnitude = signal_norms[oc] * image_norms[ic];
            if settles(value, factor * magnitude) {
                *slot = value as f32;
                continue;
            }
            // Under `2^53` of the two channels' grains, no run took a
            // rounding and no carry an error: `value` is the exact sum.
            let grains = signal_grains
                .get_or_init(|| channel_grains(source, batch, out_channels, ohw))[oc]
                * image_grains
                    .get_or_init(|| channel_grains(image, batch, in_channels, input_plane))[ic];
            *slot = if magnitude * margin < WHOLE * grains {
                (value + 0.0) as f32
            } else {
                weight_entry(oc, k, image, source, geometry)
            };
        }
    });
    gradient
}

/// The grain of each channel of a `[N, C, P]` signal over every image.
fn channel_grains(values: &[f32], batch: usize, channels: usize, plane: usize) -> Vec<f64> {
    let mut grains = vec![0.0f64; channels];
    par_out_chunks(&mut grains, 1, &|channel, slot| {
        slot[0] = (0..batch)
            .map(|n| {
                grain_of(
                    values[(n * channels + channel) * plane..][..plane]
                        .iter()
                        .copied(),
                )
            })
            .fold(f64::INFINITY, f64::min);
    });
    grains
}

/// The 2-norm of each channel of a `[N, C, P]` signal over every image: the
/// root of the float64 sum of its squares, each exact.
fn channel_norms(values: &[f32], batch: usize, channels: usize, plane: usize) -> Vec<f64> {
    let mut norms = vec![0.0f64; channels];
    par_out_chunks(&mut norms, 1, &|channel, slot| {
        let mut lanes = [0.0f64; LANES];
        for n in 0..batch {
            let run = &values[(n * channels + channel) * plane..][..plane];
            let (blocks, rest) = run.as_chunks::<LANES>();
            for block in blocks {
                for lane in 0..LANES {
                    lanes[lane] += f64::from(block[lane]) * f64::from(block[lane]);
                }
            }
            for (lane, &value) in rest.iter().enumerate() {
                lanes[lane] += f64::from(value) * f64::from(value);
            }
        }
        slot[0] = lanes.iter().sum::<f64>().sqrt();
    });
    norms
}

/// Kernel slice `index = oc * (in_channels / groups) + local` of
/// [`weight_gradient`] straight from the operands: each tap's products added
/// in float64 lanes a run at a time, each run carried into `sums` by a
/// two-sum, its rounding into `errors`.
fn weight_slice_sums(
    sums: &mut [f64],
    errors: &mut [f64],
    index: usize,
    image: &[f32],
    source: &[f32],
    geometry: &ConvGeometry,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { weight_slice_sums_avx2(sums, errors, index, image, source, geometry) };
    }
    weight_slice_sums_body::<false>(sums, errors, index, image, source, geometry)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn weight_slice_sums_avx2(
    sums: &mut [f64],
    errors: &mut [f64],
    index: usize,
    image: &[f32],
    source: &[f32],
    geometry: &ConvGeometry,
) {
    weight_slice_sums_body::<true>(sums, errors, index, image, source, geometry)
}

/// A run's lanes added together and carried into `sum` and `error`.
#[inline(always)]
fn carry_run(lanes: &mut [f64; LANES], sum: &mut f64, error: &mut f64) {
    let run = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    *lanes = [0.0; LANES];
    let (total, rounding) = two_sum(*sum, run);
    *sum = total;
    *error += rounding;
}

#[inline(always)]
fn weight_slice_sums_body<const FMA: bool>(
    sums: &mut [f64],
    errors: &mut [f64],
    index: usize,
    image: &[f32],
    source: &[f32],
    geometry: &ConvGeometry,
) {
    let ConvGeometry {
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        groups,
    } = *geometry;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let (oc, local) = (index / group_in, index % group_in);
    let ic = (oc / group_out) * group_in + local;
    let plane_in = input_height * input_width;
    let plane_out = output_height * output_width;
    let taps = kernel_h * kernel_w;
    let mut lanes = vec![[0.0f64; LANES]; taps];
    let mut counts = vec![0usize; taps];
    // Each image's two planes widened once for all of their taps.
    let mut signal = vec![0.0f64; plane_out];
    let mut picture = vec![0.0f64; plane_in];
    for n in 0..batch_size {
        let widen = |wide: &mut [f64], values: &[f32]| {
            for (slot, &value) in wide.iter_mut().zip(values) {
                *slot = f64::from(value);
            }
        };
        widen(
            &mut signal,
            &source[(n * out_channels + oc) * plane_out..][..plane_out],
        );
        widen(
            &mut picture,
            &image[(n * in_channels + ic) * plane_in..][..plane_in],
        );
        for ky in 0..kernel_h {
            let ky_off = ky * dilation.0;
            let (oh_lo, oh_hi) =
                in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
            for kx in 0..kernel_w {
                let kx_off = kx * dilation.1;
                let (ow_lo, ow_hi) =
                    in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
                if oh_lo >= oh_hi || ow_lo >= ow_hi {
                    continue;
                }
                let tap = ky * kernel_w + kx;
                let acc = &mut lanes[tap];
                for oh in oh_lo..oh_hi {
                    let from = &signal[oh * output_width + ow_lo..oh * output_width + ow_hi];
                    let row = &picture[(oh * stride.0 + ky_off - padding.0) * input_width..]
                        [..input_width];
                    if stride.1 == 1 {
                        let start = ow_lo + kx_off - padding.1;
                        let read = &row[start..start + from.len()];
                        let (g_blocks, g_tail) = from.as_chunks::<LANES>();
                        let (x_blocks, x_tail) = read.as_chunks::<LANES>();
                        for (g, x) in g_blocks.iter().zip(x_blocks) {
                            for lane in 0..LANES {
                                acc[lane] = add_exact::<FMA>(acc[lane], g[lane], x[lane]);
                            }
                        }
                        for (lane, (&g, &x)) in g_tail.iter().zip(x_tail).enumerate() {
                            acc[lane] = add_exact::<FMA>(acc[lane], g, x);
                        }
                    } else {
                        for (i, &g) in from.iter().enumerate() {
                            let x = row[(ow_lo + i) * stride.1 + kx_off - padding.1];
                            acc[i % LANES] = add_exact::<FMA>(acc[i % LANES], g, x);
                        }
                    }
                    // The first lane takes the most of a row's terms.
                    counts[tap] += from.len().div_ceil(LANES);
                    if counts[tap] >= DEPTH {
                        carry_run(acc, &mut sums[tap], &mut errors[tap]);
                        counts[tap] = 0;
                    }
                }
            }
        }
    }
    for (tap, acc) in lanes.iter_mut().enumerate() {
        carry_run(acc, &mut sums[tap], &mut errors[tap]);
    }
}

/// [`weight_gradient`] through the lowered columns, a block of images at a
/// time as the float64 path takes them: the sum of each entry and the error
/// carried beside it, `[C_out, group_k]`, and how many carries each took.
///
/// A task takes a band of up to [`BAND`] of a group's lowered rows and a
/// range of the block's positions, and multiplies them by the signal a run
/// of [`RUN`] positions at a time, widening both into buffers of its own:
/// each run's product is carried into the task's sums by a two-sum, its
/// rounding into the task's errors, and the tasks' totals are carried into
/// the whole in order.
fn lowered_weight_sums(
    image: &[f32],
    source: &[f32],
    geometry: &ConvGeometry,
) -> (Vec<f64>, Vec<f64>, usize) {
    let ConvGeometry {
        batch_size: batch,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        groups,
        ..
    } = *geometry;
    let ohw = output_height * output_width;
    let kh_kw = kernel_h * kernel_w;
    let k_dim = in_channels * kh_kw;
    let group_out = out_channels / groups;
    let group_k = (in_channels / groups) * kh_kw;
    let span_image = out_channels * ohw;
    let images = images_per_block::<f32>(batch, k_dim * ohw, span_image);
    let mut columns = vec![0.0f32; images * ohw * k_dim];
    let mut signal = vec![0.0f32; images * span_image];
    // `[groups, group_k, group_out]`: a band of a group's rows is a run.
    let mut sums = vec![0.0f64; groups * group_k * group_out];
    let mut errors = vec![0.0f64; groups * group_k * group_out];
    let threads = crate::parallel::current_num_threads().max(1);
    let bands = group_k.div_ceil(BAND);
    let mut carries = 0;

    let mut first = 0;
    while first < batch {
        let last = (first + images).min(batch);
        let span = (last - first) * ohw;
        super::block_channel_major::<f32>(
            source,
            &mut signal[..out_channels * span],
            first,
            last,
            out_channels,
            ohw,
        );
        lower_positions::<f32>(image, &mut columns, first * ohw, last * ohw, geometry);
        let (block, lowered) = (&signal[..out_channels * span], &columns[..k_dim * span]);
        let runs = span.div_ceil(RUN);
        let ranges = (2 * threads).div_ceil(bands).clamp(1, runs);
        carries += runs + ranges;
        for g in 0..groups {
            let totals = par_map_indexed(bands * ranges, &|task| {
                let (band, range) = (task / ranges, task % ranges);
                let k0 = band * BAND;
                let rows = BAND.min(group_k - k0);
                let mut hi = vec![0.0f64; rows * group_out];
                let mut lo = vec![0.0f64; rows * group_out];
                let mut left = vec![0.0f64; rows * RUN];
                let mut right = vec![0.0f64; group_out * RUN];
                let mut partial = vec![0.0f64; rows * group_out];
                for run in range * runs / ranges..(range + 1) * runs / ranges {
                    let from = run * RUN;
                    let len = RUN.min(span - from);
                    for (r, line) in left.chunks_exact_mut(len).take(rows).enumerate() {
                        let read = &lowered[(g * group_k + k0 + r) * span + from..][..len];
                        for (slot, &value) in line.iter_mut().zip(read) {
                            *slot = f64::from(value);
                        }
                    }
                    for (o, line) in right.chunks_exact_mut(len).take(group_out).enumerate() {
                        let read = &block[(g * group_out + o) * span + from..][..len];
                        for (slot, &value) in line.iter_mut().zip(read) {
                            *slot = f64::from(value);
                        }
                    }
                    // SAFETY: `left` holds `[rows, len]`, `right` `[group_out,
                    // len]` -- the logical `(len, group_out)` operand read
                    // transposed -- and `partial` `[rows, group_out]`.
                    unsafe {
                        crate::ops::linalg::gemm_nt_f64(
                            rows,
                            len,
                            group_out,
                            left.as_ptr(),
                            right.as_ptr(),
                            partial.as_mut_ptr(),
                        );
                    }
                    for ((sum, error), &value) in hi.iter_mut().zip(lo.iter_mut()).zip(&partial) {
                        let (total, rounding) = two_sum(*sum, value);
                        *sum = total;
                        *error += rounding;
                    }
                }
                (hi, lo)
            });
            let entries = g * group_k * group_out..(g + 1) * group_k * group_out;
            par_out_chunks2(
                &mut sums[entries.clone()],
                &mut errors[entries],
                BAND * group_out,
                &|start, sums, errors| {
                    let band = start / (BAND * group_out);
                    for (hi, lo) in &totals[band * ranges..(band + 1) * ranges] {
                        for (((sum, error), &high), &low) in
                            sums.iter_mut().zip(errors.iter_mut()).zip(hi).zip(lo)
                        {
                            let (total, rounding) = two_sum(*sum, high);
                            *sum = total;
                            *error += rounding + low;
                        }
                    }
                },
            );
        }
        first = last;
    }

    // Laid out as the kernel is, `[C_out, group_k]`.
    let mut by_kernel = (vec![0.0f64; sums.len()], vec![0.0f64; sums.len()]);
    for (index, (sum, error)) in by_kernel.0.iter_mut().zip(&mut by_kernel.1).enumerate() {
        let (oc, k) = (index / group_k, index % group_k);
        let (g, o) = (oc / group_out, oc % group_out);
        let from = (g * group_k + k) * group_out + o;
        *sum = sums[from];
        *error = errors[from];
    }
    (by_kernel.0, by_kernel.1, carries)
}

/// The float32 nearest entry `k` of output channel `oc`'s kernel slice in
/// [`weight_gradient`], from its own products.
fn weight_entry(
    oc: usize,
    k: usize,
    image: &[f32],
    source: &[f32],
    geometry: &ConvGeometry,
) -> f32 {
    let ConvGeometry {
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride,
        padding,
        dilation,
        groups,
    } = *geometry;
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let kh_kw = kernel_h * kernel_w;
    let (local, tap) = (k / kh_kw, k % kh_kw);
    let ky_off = (tap / kernel_w) * dilation.0;
    let kx_off = (tap % kernel_w) * dilation.1;
    let ic = (oc / group_out) * group_in + local;
    let (oh_lo, oh_hi) = in_bounds_range(ky_off, padding.0, input_height, stride.0, output_height);
    let (ow_lo, ow_hi) = in_bounds_range(kx_off, padding.1, input_width, stride.1, output_width);
    let span = ow_hi - ow_lo;
    let (mut signal, mut reads) = (Vec::new(), Vec::new());
    for n in 0..batch_size {
        for oh in oh_lo..oh_hi {
            let ih = oh * stride.0 + ky_off - padding.0;
            let from = ((n * out_channels + oc) * output_height + oh) * output_width + ow_lo;
            signal.extend_from_slice(&source[from..from + span]);
            let row =
                &image[((n * in_channels + ic) * input_height + ih) * input_width..][..input_width];
            reads.extend((ow_lo..ow_hi).map(|ow| row[ow * stride.1 + kx_off - padding.1]));
        }
    }
    dot_f32(&signal, &reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values spread over twelve binades, a few of them whole.
    fn values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|index| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let unit = (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                if index % 11 == 0 {
                    (unit * 16.0).round()
                } else {
                    unit * 2f32.powi((z % 12) as i32 - 6)
                }
            })
            .collect()
    }

    fn geometry(
        (batch, channels, height, width): (usize, usize, usize, usize),
        (outputs, kernel): (usize, (usize, usize)),
        (stride, padding, dilation): ((usize, usize), (usize, usize), (usize, usize)),
        groups: usize,
    ) -> ConvGeometry {
        let span = |k: usize, d: usize| d * (k - 1) + 1;
        ConvGeometry {
            batch_size: batch,
            in_channels: channels,
            input_height: height,
            input_width: width,
            out_channels: outputs,
            kernel_h: kernel.0,
            kernel_w: kernel.1,
            output_height: (height + 2 * padding.0 - span(kernel.0, dilation.0)) / stride.0 + 1,
            output_width: (width + 2 * padding.1 - span(kernel.1, dilation.1)) / stride.1 + 1,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Where tap `k` of output `o` reads the input, written from the
    /// definition rather than from the kernels' ranges.
    fn read_at(
        o: usize,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        extent: usize,
    ) -> Option<usize> {
        let at = (o * stride + k * dilation) as isize - padding as isize;
        (0..extent as isize).contains(&at).then_some(at as usize)
    }

    /// Every term of the convolution, as `(output, input, tap)` indices.
    fn terms(g: &ConvGeometry) -> Vec<(usize, usize, usize)> {
        let group_in = g.in_channels / g.groups;
        let group_out = g.out_channels / g.groups;
        let mut all = Vec::new();
        for n in 0..g.batch_size {
            for oc in 0..g.out_channels {
                for oh in 0..g.output_height {
                    for ow in 0..g.output_width {
                        for local in 0..group_in {
                            let ic = (oc / group_out) * group_in + local;
                            for ky in 0..g.kernel_h {
                                for kx in 0..g.kernel_w {
                                    let (Some(ih), Some(iw)) = (
                                        read_at(
                                            oh,
                                            ky,
                                            g.stride.0,
                                            g.padding.0,
                                            g.dilation.0,
                                            g.input_height,
                                        ),
                                        read_at(
                                            ow,
                                            kx,
                                            g.stride.1,
                                            g.padding.1,
                                            g.dilation.1,
                                            g.input_width,
                                        ),
                                    ) else {
                                        continue;
                                    };
                                    let output = ((n * g.out_channels + oc) * g.output_height + oh)
                                        * g.output_width
                                        + ow;
                                    let input = ((n * g.in_channels + ic) * g.input_height + ih)
                                        * g.input_width
                                        + iw;
                                    let tap = ((oc * group_in + local) * g.kernel_h + ky)
                                        * g.kernel_w
                                        + kx;
                                    all.push((output, input, tap));
                                }
                            }
                        }
                    }
                }
            }
        }
        all
    }

    /// Each of `count` sums of the products `pairs` files under it, plus
    /// `addend(index)`, rounded once.
    fn exact_sums(
        count: usize,
        pairs: impl Iterator<Item = (usize, f32, f32)>,
        addend: impl Fn(usize) -> f32,
    ) -> Vec<f32> {
        let mut lists = vec![(Vec::new(), Vec::new()); count];
        for (index, a, b) in pairs {
            lists[index].0.push(a);
            lists[index].1.push(b);
        }
        lists
            .into_iter()
            .enumerate()
            .map(|(index, (mut a, mut b))| {
                a.push(addend(index));
                b.push(1.0);
                dot_f32(&a, &b)
            })
            .collect()
    }

    fn check(g: &ConvGeometry, seed: u64) {
        let input_len = g.batch_size * g.in_channels * g.input_height * g.input_width;
        let output_len = g.batch_size * g.out_channels * g.output_height * g.output_width;
        let weight_len = g.out_channels * (g.in_channels / g.groups) * g.kernel_h * g.kernel_w;
        let input = values(seed, input_len);
        let weight = values(seed + 1, weight_len);
        let signal = values(seed + 2, output_len);
        let out_bias = values(seed + 3, g.out_channels);
        let in_bias = values(seed + 4, g.in_channels);
        let all = terms(g);
        let output_plane = g.output_height * g.output_width;
        let input_plane = g.input_height * g.input_width;

        let want = exact_sums(
            output_len,
            all.iter().map(|&(o, i, t)| (o, weight[t], input[i])),
            |o| out_bias[o / output_plane % g.out_channels],
        );
        let got = convolve(&input, &weight, Some(&out_bias), g).unwrap();
        assert_eq!(bits(&got), bits(&want), "convolution of {g:?}");
        let want = exact_sums(
            output_len,
            all.iter().map(|&(o, i, t)| (o, weight[t], input[i])),
            |_| 0.0,
        );
        let got = convolve(&input, &weight, None, g).unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "convolution of {g:?} without a bias"
        );

        let want = exact_sums(
            input_len,
            all.iter().map(|&(o, i, t)| (i, weight[t], signal[o])),
            |i| in_bias[i / input_plane % g.in_channels],
        );
        assert_eq!(
            bits(&scatter(&signal, &weight, Some(&in_bias), g)),
            bits(&want),
            "scatter of {g:?}"
        );

        let want = exact_sums(
            weight_len,
            all.iter().map(|&(o, i, t)| (t, signal[o], input[i])),
            |_| 0.0,
        );
        assert_eq!(
            bits(&weight_gradient(&input, &signal, g)),
            bits(&want),
            "weight gradient of {g:?}"
        );
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn every_entry_is_its_exact_sum_on_either_path() {
        let cases = [
            // Depthwise, and few outputs: straight from the operands.
            geometry((2, 6, 9, 11), (6, (3, 3)), ((1, 1), (1, 1), (1, 1)), 6),
            geometry((2, 3, 10, 16), (2, (3, 3)), ((1, 1), (2, 2), (2, 2)), 1),
            geometry((1, 17, 9, 9), (3, (3, 3)), ((1, 2), (1, 1), (1, 1)), 1),
            // Many outputs per group: through the lowered columns.
            geometry((2, 6, 9, 11), (48, (3, 3)), ((1, 1), (1, 1), (1, 1)), 3),
            geometry((2, 3, 13, 12), (8, (3, 3)), ((2, 2), (1, 0), (1, 1)), 1),
            geometry((2, 3, 10, 16), (20, (3, 2)), ((1, 1), (2, 2), (2, 2)), 1),
            geometry((3, 40, 5, 6), (24, (3, 3)), ((1, 1), (1, 1), (1, 1)), 1),
            // Lowered with a single output channel, and with a single output
            // position: a product of one row, and of one column.
            geometry((2, 17, 9, 9), (1, (3, 3)), ((1, 2), (1, 1), (1, 1)), 1),
            geometry((1, 40, 3, 3), (24, (3, 3)), ((1, 1), (0, 0), (1, 1)), 1),
        ];
        for (index, case) in cases.iter().enumerate() {
            check(case, 17 + 7 * index as u64);
        }
    }

    #[test]
    fn ties_zeros_and_special_values() {
        let g = geometry((2, 2, 6, 7), (2, (3, 3)), ((1, 1), (1, 1), (1, 1)), 2);
        // A kernel of eighths over whole numbers: sums on halfway points
        // past 2^24 and exact zeros, both settled from the sum itself.
        let input: Vec<f32> = (0..2 * 2 * 42)
            .map(|i| ((i * 7) % 5) as f32 - 2.0)
            .collect();
        let mut weight = vec![0.125f32; 18];
        weight[4] = -1.0;
        let bias = [2f32.powi(24) + 2.0, 0.0];
        let got = convolve(&input, &weight, Some(&bias), &g).unwrap();
        let all = terms(&g);
        let want = exact_sums(
            got.len(),
            all.iter().map(|&(o, i, t)| (o, weight[t], input[i])),
            |o| bias[o / 42 % 2],
        );
        assert_eq!(bits(&got), bits(&want));
        assert!(
            got.iter()
                .all(|value| value.to_bits() != (-0.0f32).to_bits())
        );

        // An infinity and a NaN reach only the entries that read them.
        let mut input = vec![1.0f32; 2 * 2 * 42];
        input[10] = f32::INFINITY;
        input[42 + 30] = f32::NAN;
        let got = convolve(&input, &[1.0; 18], None, &g).unwrap();
        for (output, value) in got.iter().enumerate() {
            let reads: Vec<usize> = all.iter().filter(|t| t.0 == output).map(|t| t.1).collect();
            let expect_nan = reads.contains(&(42 + 30));
            let expect_inf = reads.contains(&10);
            assert_eq!(value.is_nan(), expect_nan, "entry {output}");
            assert_eq!(
                value.is_infinite(),
                expect_inf && !expect_nan,
                "entry {output}"
            );
        }
    }
}
