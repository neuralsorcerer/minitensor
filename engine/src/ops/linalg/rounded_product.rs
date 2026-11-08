// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float32 matrix products whose every entry is the float32 nearest its exact
//! dot.
//!
//! A product of two float32s is exact in float64, so a float64 product of the
//! operands widened differs from the exact one only by the roundings of its
//! sums: under `γ_k = k·u / (1 - k·u)` of `Σ |a_l · b_l|` for an entry of `k`
//! terms, whatever order they were added in, with `u = 2^-53`. That sum is at
//! most the product of the row's and the column's 2-norms. Where both ends of
//! the bound narrow to one float32, rounding being monotonic, so does the
//! exact entry; where they do not -- one entry in some thousands of ordinary
//! data, more where an entry cancels far below its terms -- the entry is the
//! exact dot of its row and column. A product with a single row or column is
//! those dots throughout.
//!
//! Every value of a run is a whole multiple of the least power of two any of
//! them has a bit at -- its grain -- so every product of a row and a column,
//! and every partial sum of them, is a whole multiple of the product of
//! their grains. Where the norms leave the sums under `2^53` of those, every
//! one of them is a float64, and the float64 entry is the exact one: small
//! whole numbers, one-hot rows and an operand of zeros among them, whose
//! entries the bound alone would leave open.

use crate::error::{MinitensorError, Result};
use crate::ops::arithmetic::exact_sum::dot_f32;
use crate::ops::provider::Storage;

/// Float64's unit roundoff, `2^-53`.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// Rows and runs a transposed copy moves at a time, through a buffer on the
/// stack.
const TILE: usize = 32;

/// A pass over fewer elements than this runs on the calling thread, where the
/// pool's round trip would cost more than it spreads: each pass is a light
/// one, and is judged by its own length rather than the product's.
const SERIAL_BELOW: usize = 1 << 16;

/// Elements a task of the parallel passes takes, at least.
const TASK: usize = 1 << 14;

/// Columns a task of [`column_norms`] sums down.
const BAND: usize = 256;

/// `out[b] = lhs[b] @ rhs[b]` for each of `batch` products of `(m, k)` by
/// `(k, n)`, either operand read in the layout it is stored in, every entry
/// the float32 nearest its exact dot. `out` need not arrive zeroed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rounded_products_f32(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_storage: Storage,
    rhs: &[f32],
    rhs_storage: Storage,
    out: &mut [f32],
) -> Result<()> {
    if lhs.len() != batch * m * k || rhs.len() != batch * k * n || out.len() != batch * m * n {
        return Err(MinitensorError::internal_error(format!(
            "product buffers hold {}, {} and {} elements where {batch} products of ({m}, {k}) \
             by ({k}, {n}) need {}, {} and {}",
            lhs.len(),
            rhs.len(),
            out.len(),
            batch * m * k,
            batch * k * n,
            batch * m * n
        )));
    }
    if out.is_empty() {
        return Ok(());
    }
    if k == 0 {
        out.fill(0.0);
        return Ok(());
    }
    if m == 1 || n == 1 {
        return single_rows_or_columns(batch, m, k, n, lhs, lhs_storage, rhs, rhs_storage, out);
    }

    let serial = |len: usize| len < SERIAL_BELOW;
    // Both operands widened as they lie -- but for a pair both transposed,
    // which no kernel takes, whose left one is laid upright -- and the norm of
    // each row of the left operand and each column of the right, taken across
    // or down their storage.
    let (wide_lhs, gemm_lhs_storage) = match (lhs_storage, rhs_storage) {
        (Storage::Transposed, Storage::Transposed) => {
            (upright(lhs, k, m, serial(lhs.len())), Storage::RowMajor)
        }
        _ => (widened(lhs, serial(lhs.len())), lhs_storage),
    };
    let wide_rhs = widened(rhs, serial(rhs.len()));
    let lhs_norms = match lhs_storage {
        Storage::RowMajor => row_norms(lhs, k, serial(lhs.len())),
        Storage::Transposed => column_norms(lhs, k, m, serial(lhs.len())),
    };
    let rhs_norms = match rhs_storage {
        Storage::RowMajor => column_norms(rhs, k, n, serial(rhs.len())),
        Storage::Transposed => row_norms(rhs, k, serial(rhs.len())),
    };
    let mut wide = vec![0.0f64; batch * m * n];
    super::stored_gemm_f64(
        batch,
        m,
        k,
        n,
        &wide_lhs,
        gemm_lhs_storage,
        &wide_rhs,
        rhs_storage,
        &mut wide,
    )?;

    // `γ_k` with room for the norms' own roundings, under `(k + 1)u` of the
    // squares they sum and one of the root: `(1 + 4k·u)` covers their effect
    // on the product of the two, and a further `2^-40` the roundings of the
    // bound worked out from it.
    let reach = k as f64 * UNIT;
    let scale = reach / (1.0 - reach) * (1.0 + 4.0 * reach + 2f64.powi(-40));
    // Sums whose norms stay under this many grains of their row and column
    // took no rounding: `2^53`, short by the norms' own roundings. A column's
    // grain is found the first time one of its entries is left open, and
    // kept for the rest.
    let whole = 2f64.powi(53) * (1.0 - 2f64.powi(-18));
    let column_grains: Vec<std::sync::OnceLock<f64>> =
        (0..batch * n).map(|_| std::sync::OnceLock::new()).collect();
    let decide = |start: usize, part: &mut [f32]| {
        for (offset, row) in part.chunks_mut(n).enumerate() {
            let index = start / n + offset;
            let (product, i) = (index / m, index % m);
            let values = &wide[index * n..(index + 1) * n];
            let norms = &rhs_norms[product * n..(product + 1) * n];
            let row_scale = scale * lhs_norms[index];
            // Every entry narrowed, and whether any is left open, in a pass
            // the compiler can take in vector lanes; the open ones, one in
            // some thousands, are taken again below.
            let mut open = false;
            for ((slot, &value), &norm) in row.iter_mut().zip(values).zip(norms) {
                *slot = value as f32;
                open |= !settles(value, row_scale * norm);
            }
            if !open {
                continue;
            }
            let (left, right) = (&lhs[product * m * k..], &rhs[product * k * n..]);
            let lhs_row: Vec<f32> = (0..k)
                .map(|l| left[lhs_at(lhs_storage, m, k, i, l)])
                .collect();
            let row_grain = whole * grain_of(lhs_row.iter().copied());
            let mut column = Vec::with_capacity(k);
            for (j, slot) in row.iter_mut().enumerate() {
                if settles(values[j], row_scale * norms[j]) {
                    continue;
                }
                let column_grain = *column_grains[product * n + j].get_or_init(|| {
                    grain_of((0..k).map(|l| right[rhs_at(rhs_storage, k, n, l, j)]))
                });
                // A sum that took no rounding is the exact entry, a zero a
                // positive one, as a dot's is.
                if lhs_norms[index] * norms[j] < row_grain * column_grain {
                    *slot = values[j] as f32 + 0.0;
                    continue;
                }
                column.clear();
                column.extend((0..k).map(|l| right[rhs_at(rhs_storage, k, n, l, j)]));
                *slot = dot_f32(&lhs_row, &column);
            }
        }
    };
    if serial(out.len()) {
        decide(0, out);
    } else {
        let rows_per_task = (TASK / n).max(1);
        crate::ops::map::par_out_chunks(out, rows_per_task * n, &decide);
    }
    Ok(())
}

/// Whether every number within `bound` of `value` narrows to the float32
/// `value` does, by both ends narrowing alike bit for bit -- so a bound
/// reaching across zero, with a zero of either sign at its ends, settles
/// nothing. A zero or a non-finite value is left to the exact dot, whose
/// sign and special values are the rule everywhere else a dot is taken.
#[inline(always)]
fn settles(value: f64, bound: f64) -> bool {
    // The bound and the roundings of `value ± reach` both inside `reach`.
    let reach = (bound + value.abs() * 2f64.powi(-51)) * (1.0 + 2f64.powi(-20));
    let low = (value - reach) as f32;
    let high = (value + reach) as f32;
    (low.to_bits() == high.to_bits()) & (value != 0.0) & reach.is_finite()
}

/// Where element `(i, l)` of a logical `(m, k)` left operand lies in its
/// storage.
#[inline(always)]
fn lhs_at(storage: Storage, m: usize, k: usize, i: usize, l: usize) -> usize {
    match storage {
        Storage::RowMajor => i * k + l,
        Storage::Transposed => l * m + i,
    }
}

/// Where element `(l, j)` of a logical `(k, n)` right operand lies in its
/// storage.
#[inline(always)]
fn rhs_at(storage: Storage, k: usize, n: usize, l: usize, j: usize) -> usize {
    match storage {
        Storage::RowMajor => l * n + j,
        Storage::Transposed => j * k + l,
    }
}

/// `values` widened to float64, in their layout.
fn widened(values: &[f32], serial: bool) -> Vec<f64> {
    if serial {
        values.iter().map(|&value| f64::from(value)).collect()
    } else {
        crate::ops::map::unary_map(values, f64::from)
    }
}

/// The least power of two `value` is a whole multiple of -- the value of its
/// lowest bit -- and infinity for a zero, a multiple of every one. An
/// infinity's is infinite too, and a NaN's a NaN, but their norms settle
/// nothing.
///
/// The lowest bit of the significand cleared, the difference from the value
/// is exact, both lying in one binade; a power of two is its own. Float and
/// integer steps the compiler takes in vector lanes, where reading the bit's
/// place from a count of trailing zeros took eight times as long.
#[inline(always)]
fn grain(value: f32) -> f32 {
    let bits = value.to_bits() & 0x7fff_ffff;
    let magnitude = f32::from_bits(bits);
    let cleared = f32::from_bits(bits & bits.wrapping_sub(1));
    let power = bits & 0x7f_ffff == 0;
    let lowest = if power {
        magnitude
    } else {
        magnitude - cleared
    };
    if bits == 0 { f32::INFINITY } else { lowest }
}

/// The least of the [`grain`]s of `values`: the greatest power of two every
/// one of them is a whole multiple of.
fn grain_of(values: impl Iterator<Item = f32>) -> f64 {
    f64::from(values.map(grain).fold(f32::INFINITY, f32::min))
}

/// The 2-norm of each run of `len` in `values`: the root of the float64 sum
/// of its squares, each exact.
fn row_norms(values: &[f32], len: usize, serial: bool) -> Vec<f64> {
    let norm = |run: &[f32]| {
        let mut lanes = [0.0f64; 8];
        let (chunks, rest) = run.as_chunks::<8>();
        for chunk in chunks {
            for lane in 0..8 {
                let value = f64::from(chunk[lane]);
                lanes[lane] += value * value;
            }
        }
        let tail: f64 = rest
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        (lanes.iter().sum::<f64>() + tail).sqrt()
    };
    let mut out = vec![0.0f64; values.len() / len];
    let work = |first: usize, part: &mut [f64]| {
        for (offset, slot) in part.iter_mut().enumerate() {
            *slot = norm(&values[(first + offset) * len..(first + offset + 1) * len]);
        }
    };
    if serial {
        work(0, &mut out);
    } else {
        crate::ops::map::par_out_chunks(&mut out, (TASK / len).max(1), &work);
    }
    out
}

/// [`row_norms`] of each column of stacks of `(rows, cols)` matrices in
/// `values`, stack after stack: a band of columns summed down its stack's
/// rows at a time.
fn column_norms(values: &[f32], rows: usize, cols: usize, serial: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len() / rows];
    let work = |first: usize, part: &mut [f64]| {
        // A band may run from one stack into the next.
        let mut done = 0;
        while done < part.len() {
            let (stack, column) = ((first + done) / cols, (first + done) % cols);
            let width = (cols - column).min(part.len() - done);
            let sums = &mut part[done..done + width];
            let matrix = &values[stack * rows * cols..(stack + 1) * rows * cols];
            for row in matrix.chunks_exact(cols) {
                for (sum, &value) in sums.iter_mut().zip(&row[column..column + width]) {
                    *sum += f64::from(value) * f64::from(value);
                }
            }
            for sum in sums.iter_mut() {
                *sum = sum.sqrt();
            }
            done += width;
        }
    };
    if serial {
        work(0, &mut out);
    } else {
        crate::ops::map::par_out_chunks(&mut out, BAND, &work);
    }
    out
}

/// Stacks of `(rows, cols)` matrices laid upright as `(cols, rows)`, widened
/// to float64: a block of `TILE` by `TILE` at a time through a buffer on the
/// stack, so each line of the source is read once and each of the copy
/// written once.
fn upright(values: &[f32], rows: usize, cols: usize, serial: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    let work = |first: usize, part: &mut [f64]| {
        // Whole output rows -- source columns -- from `first`.
        let first_run = first / rows;
        let mut run = 0;
        let runs = part.len() / rows;
        while run < runs {
            let (stack, column) = ((first_run + run) / cols, (first_run + run) % cols);
            let width = TILE.min(runs - run).min(cols - column);
            let source = &values[stack * rows * cols..(stack + 1) * rows * cols];
            let mut block = [[0.0f32; TILE]; TILE];
            for top in (0..rows).step_by(TILE) {
                let height = TILE.min(rows - top);
                for (line, row) in block.iter_mut().zip(top..top + height) {
                    let at = row * cols + column;
                    line[..width].copy_from_slice(&source[at..at + width]);
                }
                for across in 0..width {
                    let at = (run + across) * rows + top;
                    for (slot, line) in part[at..at + height].iter_mut().zip(&block) {
                        *slot = f64::from(line[across]);
                    }
                }
            }
            run += width;
        }
    };
    if serial {
        work(0, &mut out);
    } else {
        crate::ops::map::par_out_chunks(
            &mut out,
            (TASK / rows).max(1).next_multiple_of(TILE) * rows,
            &work,
        );
    }
    out
}

/// A product whose output is a single row or a single column: each entry
/// the exact dot of its row and column, through the kernels that take one.
#[allow(clippy::too_many_arguments)]
fn single_rows_or_columns(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_storage: Storage,
    rhs: &[f32],
    rhs_storage: Storage,
    out: &mut [f32],
) -> Result<()> {
    let (a_size, b_size, c_size) = (m * k, k * n, m * n);
    for product in 0..batch {
        let a = &lhs[product * a_size..(product + 1) * a_size];
        let b = &rhs[product * b_size..(product + 1) * b_size];
        let c = &mut out[product * c_size..(product + 1) * c_size];
        match (lhs_storage, rhs_storage) {
            // SAFETY: each pointer is to a slice of exactly the extent its
            // kernel reads or writes, checked by the caller.
            (Storage::RowMajor, Storage::RowMajor) => unsafe {
                super::gemm_f32(m, k, n, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
            },
            (Storage::RowMajor, Storage::Transposed) => unsafe {
                super::gemm_nt_f32(m, k, n, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
            },
            (Storage::Transposed, Storage::RowMajor) => unsafe {
                super::gemm_tn_f32(m, k, n, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
            },
            (Storage::Transposed, Storage::Transposed) => {
                // Both operands read across their storage: each entry gathered.
                for i in 0..m {
                    let row: Vec<f32> =
                        (0..k).map(|l| a[lhs_at(lhs_storage, m, k, i, l)]).collect();
                    for j in 0..n {
                        let column: Vec<f32> =
                            (0..k).map(|l| b[rhs_at(rhs_storage, k, n, l, j)]).collect();
                        c[i * n + j] = dot_f32(&row, &column);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                unit * if state & 7 == 0 { 1e4 } else { 1.0 }
            })
            .collect()
    }

    fn reference(
        batch: usize,
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        lhs_storage: Storage,
        rhs: &[f32],
        rhs_storage: Storage,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        for product in 0..batch {
            for i in 0..m {
                let row: Vec<f32> = (0..k)
                    .map(|l| lhs[product * m * k + lhs_at(lhs_storage, m, k, i, l)])
                    .collect();
                for j in 0..n {
                    let column: Vec<f32> = (0..k)
                        .map(|l| rhs[product * k * n + rhs_at(rhs_storage, k, n, l, j)])
                        .collect();
                    out.push(dot_f32(&row, &column));
                }
            }
        }
        out
    }

    #[test]
    fn every_entry_is_its_exact_dot_in_every_layout() {
        for (batch, m, k, n) in [
            (1, 7, 300, 5),
            (3, 16, 33, 9),
            (2, 1, 40, 6),
            (2, 5, 17, 1),
            (1, 40, 2, 70),
        ] {
            let lhs = values(11 + m as u64, batch * m * k);
            let rhs = values(29 + n as u64, batch * k * n);
            for lhs_storage in [Storage::RowMajor, Storage::Transposed] {
                for rhs_storage in [Storage::RowMajor, Storage::Transposed] {
                    let mut out = vec![f32::NAN; batch * m * n];
                    rounded_products_f32(
                        batch,
                        m,
                        k,
                        n,
                        &lhs,
                        lhs_storage,
                        &rhs,
                        rhs_storage,
                        &mut out,
                    )
                    .unwrap();
                    let want = reference(batch, (m, k, n), &lhs, lhs_storage, &rhs, rhs_storage);
                    assert_eq!(
                        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{batch} x ({m}, {k}) @ ({k}, {n}), {lhs_storage:?} by {rhs_storage:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn whole_numbers_and_zeros_need_no_exact_dot() {
        // An identity, a zero row, whole numbers and negative zeros: every
        // entry the float64 product gives is exact, a zero a positive one.
        let n = 48;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        identity[3] = -0.0;
        let mut whole: Vec<f32> = (0..n * n)
            .map(|i| ((i * 7919) % 23) as f32 - 11.0)
            .collect();
        whole[..n].fill(-0.0);
        let mut out = vec![f32::NAN; n * n];
        rounded_products_f32(
            1,
            n,
            n,
            n,
            &whole,
            Storage::RowMajor,
            &identity,
            Storage::RowMajor,
            &mut out,
        )
        .unwrap();
        let want: Vec<f32> = whole.iter().map(|&v| v + 0.0).collect();
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            out,
            reference(
                1,
                (n, n, n),
                &whole,
                Storage::RowMajor,
                &identity,
                Storage::RowMajor
            )
        );
    }

    #[test]
    fn cancellation_zeros_and_special_values() {
        // Rows whose dots cancel far below their terms, a zero row, and a
        // row with an infinity and a NaN.
        let (m, k, n) = (4, 3, 3);
        let lhs = vec![
            1e8,
            1.0,
            -1e8, //
            0.0,
            0.0,
            0.0, //
            f32::INFINITY,
            1.0,
            2.0, //
            f32::NAN,
            1.0,
            1.0,
        ];
        let rhs = vec![
            1.0, 2.0, 0.0, //
            3e-8, 1.0, 1.0, //
            1.0, 2.0, -1.0,
        ];
        let mut out = vec![0.0f32; m * n];
        rounded_products_f32(
            1,
            m,
            k,
            n,
            &lhs,
            Storage::RowMajor,
            &rhs,
            Storage::RowMajor,
            &mut out,
        )
        .unwrap();
        let want = reference(
            1,
            (m, k, n),
            &lhs,
            Storage::RowMajor,
            &rhs,
            Storage::RowMajor,
        );
        for (got, want) in out.iter().zip(&want) {
            assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
        }
        assert_eq!(out[0], 3e-8);
        assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
    }
}
