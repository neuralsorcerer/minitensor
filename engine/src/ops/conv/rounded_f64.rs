// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float64 convolutions whose every entry is the float64 nearest its exact
//! value.
//!
//! A convolution, and each of its gradients, is a set of sums of products of
//! two float64s, and float64 has no wider float to add them in. Each of the
//! three is put instead as what is already rounded once:
//!
//! * The output is lowered to columns, a block of positions at a time, and
//!   multiplied by the kernel as matrix products rounded once -- one for
//!   each group, the bias their addend.
//! * The input gradient -- a transposed convolution's output, its bias added
//!   in -- is such products as well. Along an axis, position `i` of the grid
//!   is reached by tap `t` of the kernel where `i + padding - t · dilation`
//!   is a whole number of strides, so the positions fall into as many
//!   classes as the stride, each with the taps that reach it, and a class of
//!   the two axes is a convolution without a stride of the signal by its
//!   taps: its entries are the product of those taps, a row for each channel
//!   of the grid, by the signal lowered for them.
//! * The weight gradient is, for each entry, one dot over every position of
//!   every image, of a channel of the signal with a row of the lowered
//!   image. Where every position fits one block of the lowering it is those
//!   products, rounded once; where they do not it is a product too long to
//!   hold at once, taken a block of positions at a time, each entry's total
//!   carried from block to block as `dot` carries one -- the blocks' dots
//!   themselves where a group has few output channels, and the pairs their
//!   cut operands' products make where it has more -- and an entry the
//!   carried total does not settle is `dot` of its own terms.

use super::{COL_BLOCK_BYTES, ConvGeometry, convolve_lowered, lower_positions, weight_terms};
use crate::error::{MinitensorError, Result};
use crate::ops::arithmetic::exact_sum::dot_f64;
use crate::ops::linalg::{Affine, Totals, rounded_affine_products_f64};
use crate::ops::map::par_out_chunks;
use crate::ops::provider::Storage;

/// The first `len` values of a kernel, or what to raise where it holds
/// fewer than its geometry says.
fn kernel_of(weight: &[f64], len: usize) -> Result<&[f64]> {
    weight.get(..len).ok_or_else(|| {
        MinitensorError::internal_error(format!(
            "a convolution's kernel holds {} values where its geometry takes {len}",
            weight.len()
        ))
    })
}

/// The convolution of `input` by `weight`, `bias` added, every entry the
/// float64 nearest its exact value; the shapes are those
/// [`super::ConvScalar::convolve`] takes.
pub(super) fn convolve(
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    geometry: &ConvGeometry,
) -> Result<Vec<f64>> {
    let ConvGeometry {
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        groups,
        ..
    } = *geometry;
    let group_out = out_channels / groups;
    let group_k = (in_channels / groups) * kernel_h * kernel_w;
    let kernel = kernel_of(weight, out_channels * group_k)?;
    // The groups are a batch of products, each group's bias its rows'
    // addend.
    convolve_lowered(input, geometry, &mut |width, cols, block| {
        rounded_affine_products_f64(
            groups,
            group_out,
            group_k,
            width,
            (kernel, Storage::RowMajor),
            (cols, Storage::RowMajor),
            bias.map(|bias| Affine {
                alpha: 1.0,
                beta: 1.0,
                values: bias,
                strides: [group_out, 1, 0],
            }),
            block,
        )
    })
}

/// The positions along one axis of the grid that the same taps of the
/// kernel reach, a stride apart, and those taps.
struct Class {
    /// The first of the positions, and how many there are.
    first: usize,
    count: usize,
    /// Each tap, and where in the signal it reads for the first position:
    /// for the `a`-th, `a` further on, and nothing outside the signal.
    taps: Vec<(usize, isize)>,
}

/// The classes of one axis of a grid `extent` long under a kernel of
/// `kernel` taps: position `i` and tap `t` are of one class where `i +
/// padding` and `t · dilation` leave the same remainder over the stride.
fn classes(
    extent: usize,
    kernel: usize,
    (stride, padding, dilation): (usize, usize, usize),
) -> Vec<Class> {
    (0..stride)
        .map(|class| {
            let first = (class + stride - padding % stride) % stride;
            let count = if first < extent {
                (extent - first).div_ceil(stride)
            } else {
                0
            };
            let taps = (0..kernel)
                .filter(|tap| (tap * dilation) % stride == class)
                .map(|tap| {
                    let ahead = (first + padding) as isize - (tap * dilation) as isize;
                    (tap, ahead / stride as isize)
                })
                .collect();
            Class { first, count, taps }
        })
        .collect()
}

/// Positions `[first, last)` of one class of the grid, counted image after
/// image and row after row, lowered into `cols`: a row for each channel of
/// the signal at each of the class's taps, holding what that tap reads of
/// that channel for each position, and a zero where it reads outside the
/// signal.
fn lower_class(
    source: &[f64],
    cols: &mut [f64],
    (first, last): (usize, usize),
    (rows, columns): (&Class, &Class),
    geometry: &ConvGeometry,
) {
    let ConvGeometry {
        out_channels,
        output_height,
        output_width,
        ..
    } = *geometry;
    let width = last - first;
    let taps = rows.taps.len() * columns.taps.len();
    par_out_chunks(
        &mut cols[..out_channels * taps * width],
        width,
        &|start, row| {
            row.fill(0.0);
            let index = start / width;
            let (oc, tap) = (index / taps, index % taps);
            let down = rows.taps[tap / columns.taps.len()].1;
            let along = columns.taps[tap % columns.taps.len()].1;
            // The class's columns whose reads lie inside the signal.
            let reach = columns.count as isize;
            let lo = (-along).clamp(0, reach) as usize;
            let hi = (output_width as isize - along).clamp(0, reach) as usize;
            let mut position = first;
            while position < last {
                let (line, from) = (position / columns.count, position % columns.count);
                let take = (columns.count - from).min(last - position);
                let (n, a) = (line / rows.count, line % rows.count);
                let oh = a as isize + down;
                let (begin, end) = (from.max(lo), (from + take).min(hi));
                if (0..output_height as isize).contains(&oh) && begin < end {
                    let line =
                        ((n * out_channels + oc) * output_height + oh as usize) * output_width;
                    let read = line + (begin as isize + along) as usize;
                    row[position - first + (begin - from)..][..end - begin]
                        .copy_from_slice(&source[read..read + (end - begin)]);
                }
                position += take;
            }
        },
    );
}

/// `col2im(weight^T @ source)`, `bias` added to each channel of the grid,
/// every entry the float64 nearest its exact value; the shapes are those
/// [`super::ConvScalar::scatter`] takes.
pub(super) fn scatter(
    source: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    geometry: &ConvGeometry,
) -> Result<Vec<f64>> {
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
        stride,
        padding,
        dilation,
        groups,
    } = *geometry;
    let plane = input_height * input_width;
    let mut destination = vec![0.0f64; batch * in_channels * plane];
    if destination.is_empty() {
        return Ok(destination);
    }
    if let Some(bias) = bias {
        // An entry no tap reaches is its bias -- a zero a positive one, as
        // it is where every tap that reaches it reads a zero.
        par_out_chunks(&mut destination, plane, &|start, part| {
            part.fill(bias[(start / plane) % in_channels] + 0.0);
        });
    }
    if out_channels * output_height * output_width == 0 {
        return Ok(destination);
    }
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let kernel = kernel_of(weight, out_channels * group_in * kernel_h * kernel_w)?;
    let rows_of = classes(input_height, kernel_h, (stride.0, padding.0, dilation.0));
    let columns_of = classes(input_width, kernel_w, (stride.1, padding.1, dilation.1));
    for rows in &rows_of {
        for columns in &columns_of {
            let taps = rows.taps.len() * columns.taps.len();
            let grid = rows.count * columns.count;
            if taps == 0 || grid == 0 {
                continue;
            }
            // The class's taps of the kernel, a row for each channel of the
            // grid: `[groups, group_in, group_out * taps]`, the columns in
            // the order of the lowered signal's rows.
            let class_k = group_out * taps;
            let places: Vec<usize> = rows
                .taps
                .iter()
                .flat_map(|&(ky, _)| columns.taps.iter().map(move |&(kx, _)| ky * kernel_w + kx))
                .collect();
            let mut class_kernel = vec![0.0f64; in_channels * class_k];
            par_out_chunks(&mut class_kernel, class_k, &|start, row| {
                let ic = start / class_k;
                let (g, local) = (ic / group_in, ic % group_in);
                for (o, slice) in row.chunks_exact_mut(taps).enumerate() {
                    let from = ((g * group_out + o) * group_in + local) * kernel_h * kernel_w;
                    for (slot, &place) in slice.iter_mut().zip(&places) {
                        *slot = kernel[from + place];
                    }
                }
            });
            // A block of the class's positions at a time, as the output's
            // are taken, whole rows of them where one fits.
            let lowered = out_channels * taps;
            let positions_total = batch * grid;
            let budget = (COL_BLOCK_BYTES / (lowered * 8)).clamp(1, positions_total);
            let block_n = if budget >= columns.count {
                (budget / columns.count) * columns.count
            } else {
                budget
            };
            let mut cols = vec![0.0f64; lowered * block_n];
            let mut block = vec![0.0f64; in_channels * block_n];
            let mut first = 0;
            while first < positions_total {
                let last = (first + block_n).min(positions_total);
                let width = last - first;
                lower_class(source, &mut cols, (first, last), (rows, columns), geometry);
                rounded_affine_products_f64(
                    groups,
                    group_in,
                    class_k,
                    width,
                    (&class_kernel, Storage::RowMajor),
                    (&cols[..lowered * width], Storage::RowMajor),
                    bias.map(|bias| Affine {
                        alpha: 1.0,
                        beta: 1.0,
                        values: bias,
                        strides: [group_in, 1, 0],
                    }),
                    &mut block[..in_channels * width],
                )?;

                // `[C_in, width]` onto the class's positions of each plane
                // the block touches, a plane a task.
                let first_image = first / grid;
                let last_image = (last - 1) / grid + 1;
                let region = &mut destination
                    [first_image * in_channels * plane..last_image * in_channels * plane];
                let block = &block;
                par_out_chunks(region, plane, &|start, target| {
                    let index = start / plane;
                    let (n, ic) = (first_image + index / in_channels, index % in_channels);
                    let image_start = n * grid;
                    let from = first.max(image_start) - image_start;
                    let to = last.min(image_start + grid) - image_start;
                    let mut values =
                        &block[ic * width + (image_start + from - first)..][..to - from];
                    let mut position = from;
                    while position < to {
                        let (a, b) = (position / columns.count, position % columns.count);
                        let take = (columns.count - b).min(to - position);
                        let (line, rest) = values.split_at(take);
                        let row =
                            &mut target[(rows.first + a * stride.0) * input_width..][..input_width];
                        for (offset, &value) in line.iter().enumerate() {
                            row[columns.first + (b + offset) * stride.1] = value;
                        }
                        values = rest;
                        position += take;
                    }
                });
                first = last;
            }
        }
    }
    Ok(destination)
}

/// Positions `[first, last)` of each channel of a `[N, C, P]` signal, image
/// after image, laid side by side into `out`: `[C, last - first]`.
fn lay_positions(
    source: &[f64],
    out: &mut [f64],
    (first, last): (usize, usize),
    channels: usize,
    plane: usize,
) {
    let width = last - first;
    par_out_chunks(out, width, &|start, row| {
        let channel = start / width;
        let mut position = first;
        while position < last {
            let (n, from) = (position / plane, position % plane);
            let take = (plane - from).min(last - position);
            row[position - first..][..take]
                .copy_from_slice(&source[(n * channels + channel) * plane + from..][..take]);
            position += take;
        }
    });
}

/// The least magnitude that is not a zero's in each channel of a `[N, C,
/// P]` signal over every image, and an infinity for a channel of nothing
/// but zeros.
fn least_magnitudes(values: &[f64], batch: usize, channels: usize, plane: usize) -> Vec<f64> {
    let mut least = vec![f64::INFINITY; channels];
    par_out_chunks(&mut least, 1, &|channel, slot| {
        let mut lanes = [f64::INFINITY; 4];
        let mut take = |lane: usize, value: f64| {
            let magnitude = if value == 0.0 {
                f64::INFINITY
            } else {
                value.abs()
            };
            lanes[lane] = if magnitude < lanes[lane] {
                magnitude
            } else {
                lanes[lane]
            };
        };
        for n in 0..batch {
            let (chunks, rest) =
                values[(n * channels + channel) * plane..][..plane].as_chunks::<4>();
            for chunk in chunks {
                for (lane, &value) in chunk.iter().enumerate() {
                    take(lane, value);
                }
            }
            for &value in rest {
                take(0, value);
            }
        }
        slot[0] = lanes.into_iter().fold(f64::INFINITY, f64::min);
    });
    least
}

/// `source @ im2col(image)^T`, every entry the float64 nearest its exact
/// value; the shapes are those [`super::ConvScalar::weight_gradient`]
/// takes.
pub(super) fn weight_gradient(
    image: &[f64],
    source: &[f64],
    geometry: &ConvGeometry,
) -> Result<Vec<f64>> {
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
    let k_dim = in_channels * kh_kw;
    let positions_total = batch * ohw;
    if out_channels * group_k == 0 || positions_total == 0 {
        // A sum of no products: a positive zero.
        return Ok(vec![0.0f64; out_channels * group_k]);
    }

    // A block of positions at a time, as the output's are taken: the image
    // lowered for them, a row for each entry of a kernel slice, and the
    // signal laid a channel a row beside it. A group's entries are the
    // product of its channels of the signal by its rows of the lowered
    // image, which lie across the dots, and each block is a stretch of
    // those.
    let per_column = k_dim.max(out_channels) * 8;
    let budget = (COL_BLOCK_BYTES / per_column).clamp(1, positions_total);
    let block_n = if budget >= output_width {
        (budget / output_width) * output_width
    } else {
        budget
    };
    let mut cols = vec![0.0f64; k_dim * block_n];
    let mut signal = vec![0.0f64; out_channels * block_n];
    if block_n == positions_total {
        // Every position at once: the products themselves, rounded once.
        let mut gradient = vec![0.0f64; out_channels * group_k];
        lower_positions::<f64>(image, &mut cols, 0, positions_total, geometry);
        lay_positions(source, &mut signal, (0, positions_total), out_channels, ohw);
        rounded_affine_products_f64(
            groups,
            group_out,
            positions_total,
            group_k,
            (&signal, Storage::RowMajor),
            (&cols, Storage::Transposed),
            None,
            &mut gradient,
        )?;
        return Ok(gradient);
    }
    let mut totals = Totals::new(groups, group_out, group_k);
    let mut first = 0;
    while first < positions_total {
        let last = (first + block_n).min(positions_total);
        let width = last - first;
        lower_positions::<f64>(image, &mut cols, first, last, geometry);
        lay_positions(
            source,
            &mut signal[..out_channels * width],
            (first, last),
            out_channels,
            ohw,
        );
        totals.add(
            width,
            &signal[..out_channels * width],
            &cols[..k_dim * width],
        );
        first = last;
    }

    // A row of the lowered image holds values of one channel of the image,
    // and zeros. An entry its total does not settle is `dot` of its own
    // terms.
    let plane = input_height * input_width;
    let least_signal = least_magnitudes(source, batch, out_channels, ohw);
    let least_image: Vec<f64> = least_magnitudes(image, batch, in_channels, plane)
        .into_iter()
        .flat_map(|least| std::iter::repeat_n(least, kh_kw))
        .collect();
    Ok(totals.settled((&least_signal, &least_image), &|index| {
        let (oc, k) = (index / group_k, index % group_k);
        let (signal, reads) = weight_terms(oc, k, image, source, geometry);
        dot_f64(&signal, &reads)
    }))
}

#[cfg(test)]
mod tests {
    use super::super::testing::{geometry, terms};
    use super::*;

    /// Values spread over twelve binades, a few of them whole and a few
    /// zeros.
    fn values(seed: u64, len: usize) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|index| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let unit = (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                match index % 11 {
                    0 => (unit * 16.0).round(),
                    5 if z.is_multiple_of(3) => 0.0,
                    _ => unit * 2f64.powi((z % 12) as i32 - 6),
                }
            })
            .collect()
    }

    /// Each of `count` sums of the products `pairs` files under it, plus
    /// `addend(index)`, rounded once.
    fn exact_sums(
        count: usize,
        pairs: impl Iterator<Item = (usize, f64, f64)>,
        addend: impl Fn(usize) -> f64,
    ) -> Vec<f64> {
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
                dot_f64(&a, &b)
            })
            .collect()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
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

        let forward = || all.iter().map(|&(o, i, t)| (o, weight[t], input[i]));
        let want = exact_sums(output_len, forward(), |o| {
            out_bias[o / output_plane % g.out_channels]
        });
        let got = convolve(&input, &weight, Some(&out_bias), g).unwrap();
        assert_eq!(bits(&got), bits(&want), "convolution of {g:?}");
        let want = exact_sums(output_len, forward(), |_| 0.0);
        let got = convolve(&input, &weight, None, g).unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "convolution of {g:?} without a bias"
        );

        let back = || all.iter().map(|&(o, i, t)| (i, weight[t], signal[o]));
        let want = exact_sums(input_len, back(), |i| {
            in_bias[i / input_plane % g.in_channels]
        });
        let got = scatter(&signal, &weight, Some(&in_bias), g).unwrap();
        assert_eq!(bits(&got), bits(&want), "scatter of {g:?}");
        let want = exact_sums(input_len, back(), |_| 0.0);
        let got = scatter(&signal, &weight, None, g).unwrap();
        assert_eq!(bits(&got), bits(&want), "scatter of {g:?} without a bias");

        let want = exact_sums(
            weight_len,
            all.iter().map(|&(o, i, t)| (t, signal[o], input[i])),
            |_| 0.0,
        );
        let got = weight_gradient(&input, &signal, g).unwrap();
        assert_eq!(bits(&got), bits(&want), "weight gradient of {g:?}");
    }

    #[test]
    fn every_entry_is_its_exact_sum() {
        let cases = [
            // Few output channels to a group, each taken a row at a time.
            geometry((2, 6, 9, 11), (6, (3, 3)), ((1, 1), (1, 1), (1, 1)), 6),
            geometry((2, 3, 10, 16), (2, (3, 3)), ((1, 1), (2, 2), (2, 2)), 1),
            geometry((1, 17, 9, 9), (3, (3, 3)), ((1, 2), (1, 1), (1, 1)), 1),
            // More of them, as products of cut operands.
            geometry((2, 6, 9, 11), (48, (3, 3)), ((1, 1), (1, 1), (1, 1)), 3),
            geometry((2, 3, 10, 16), (20, (3, 2)), ((1, 1), (2, 2), (2, 2)), 1),
            geometry((3, 40, 5, 6), (24, (3, 3)), ((1, 1), (1, 1), (1, 1)), 1),
            // A single output channel, and a single output position.
            geometry((2, 17, 9, 9), (1, (3, 3)), ((1, 2), (1, 1), (1, 1)), 1),
            geometry((1, 40, 3, 3), (24, (3, 3)), ((1, 1), (0, 0), (1, 1)), 1),
            // Strides: the grid's positions in classes, some of whose taps
            // are few, with a dilation that shares a factor with the stride;
            // padding past the kernel; and a stride past it, which leaves
            // positions of the grid that no tap reaches.
            geometry((2, 3, 13, 12), (8, (3, 3)), ((2, 2), (1, 0), (1, 1)), 1),
            geometry((2, 4, 11, 13), (6, (4, 3)), ((3, 2), (2, 1), (2, 3)), 1),
            geometry((1, 3, 6, 7), (5, (2, 2)), ((2, 3), (3, 4), (1, 1)), 1),
            geometry((2, 4, 11, 12), (6, (2, 2)), ((3, 4), (0, 1), (1, 1)), 2),
        ];
        for (index, case) in cases.iter().enumerate() {
            check(case, 17 + 7 * index as u64);
        }
    }

    #[test]
    fn classes_share_out_the_positions_and_the_taps() {
        // Every position of the grid and every tap of the kernel is of one
        // class, and a tap reads for a position what the definition says.
        for (extent, kernel, stride, padding, dilation) in [
            (11, 3, 1, 1, 1),
            (13, 4, 3, 2, 2),
            (7, 2, 3, 4, 1),
            (12, 5, 4, 0, 6),
            (5, 3, 7, 1, 2),
        ] {
            let all = classes(extent, kernel, (stride, padding, dilation));
            let mut positions = vec![0; extent];
            let mut taps = vec![0; kernel];
            for class in &all {
                for a in 0..class.count {
                    let i = class.first + a * stride;
                    positions[i] += 1;
                    for &(tap, ahead) in &class.taps {
                        // `o · stride + tap · dilation - padding = i`.
                        let o = ahead + a as isize;
                        assert_eq!(
                            o * stride as isize + (tap * dilation) as isize - padding as isize,
                            i as isize
                        );
                    }
                }
                for &(tap, _) in &class.taps {
                    taps[tap] += 1;
                }
            }
            assert!(positions.iter().all(|&count| count == 1));
            assert!(taps.iter().all(|&count| count == 1));
        }
    }

    #[test]
    fn zeros_whole_numbers_and_special_values() {
        let g = geometry((2, 2, 6, 7), (2, (3, 3)), ((1, 1), (1, 1), (1, 1)), 2);
        let all = terms(&g);
        // A kernel of eighths over whole numbers, with a bias past 2^53:
        // sums on halfway points and exact zeros, a zero a positive one.
        let input: Vec<f64> = (0..2 * 2 * 42)
            .map(|i| ((i * 7) % 5) as f64 - 2.0)
            .collect();
        let mut weight = vec![0.125f64; 18];
        weight[4] = -1.0;
        let bias = [2f64.powi(53) + 2.0, -0.0];
        let got = convolve(&input, &weight, Some(&bias), &g).unwrap();
        let want = exact_sums(
            got.len(),
            all.iter().map(|&(o, i, t)| (o, weight[t], input[i])),
            |o| bias[o / 42 % 2],
        );
        assert_eq!(bits(&got), bits(&want));
        let signal = vec![1.0; got.len()];
        for values in [
            got,
            scatter(&signal, &weight, Some(&bias), &g).unwrap(),
            weight_gradient(&input, &signal, &g).unwrap(),
        ] {
            assert!(values.iter().all(|v| v.to_bits() != (-0.0f64).to_bits()));
        }
        // A channel of the signal that is nothing but zeros, as a unit that
        // never fired leaves it: its slices of the kernel's gradient are.
        let mut signal = values(3, 2 * 2 * 42);
        for n in 0..2 {
            signal[(n * 2 + 1) * 42..][..42].fill(0.0);
        }
        let gradient = weight_gradient(&input, &signal, &g).unwrap();
        assert_eq!(bits(&gradient[9..]), vec![0; 9]);

        // An infinity and a NaN reach only the entries that read them.
        let mut input = vec![1.0f64; 2 * 2 * 42];
        input[10] = f64::INFINITY;
        input[42 + 30] = f64::NAN;
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

    #[test]
    fn rows_wider_than_a_block_of_the_lowering() {
        // One row of 3585 positions under 512 taps: lowered 2048 positions
        // at a time, so a block starts and ends part way along the row, and
        // the kernel's gradient is carried from one block to the next -- as
        // dots for two output channels and as pairs for fourteen. Entries
        // here and there are held against `dot` of their own terms.
        for outputs in [2, 14] {
            let g = geometry(
                (1, 1, 1, 4096),
                (outputs, (1, 512)),
                ((1, 1), (0, 0), (1, 1)),
                1,
            );
            let positions = g.output_width;
            let input = values(5, 4096);
            let weight = values(6, outputs * 512);
            let signal = values(7, outputs * positions);
            let bias = values(8, outputs);
            let got = convolve(&input, &weight, Some(&bias), &g).unwrap();
            for index in (0..got.len())
                .step_by(97)
                .chain([2047, 2048, got.len() - 1])
            {
                let (oc, at) = (index / positions, index % positions);
                let mut taps = weight[oc * 512..][..512].to_vec();
                let mut reads = input[at..at + 512].to_vec();
                taps.push(bias[oc]);
                reads.push(1.0);
                assert_eq!(got[index].to_bits(), dot_f64(&taps, &reads).to_bits());
            }
            let got = weight_gradient(&input, &signal, &g).unwrap();
            for index in (0..got.len()).step_by(41) {
                let (oc, tap) = (index / 512, index % 512);
                let want = dot_f64(
                    &signal[oc * positions..][..positions],
                    &input[tap..][..positions],
                );
                assert_eq!(got[index].to_bits(), want.to_bits(), "kernel entry {index}");
            }
            let got = scatter(&signal, &weight, None, &g).unwrap();
            for i in (0..4096).step_by(53).chain([0, 511, 512, 4095]) {
                let (mut taps, mut reads) = (Vec::new(), Vec::new());
                for oc in 0..outputs {
                    for tap in 0..512 {
                        if i >= tap && i - tap < positions {
                            taps.push(weight[oc * 512 + tap]);
                            reads.push(signal[oc * positions + i - tap]);
                        }
                    }
                }
                assert_eq!(
                    got[i].to_bits(),
                    dot_f64(&taps, &reads).to_bits(),
                    "grid {i}"
                );
            }
        }
    }
}
