// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Float64 matrix products whose every entry is the float64 nearest its exact
//! dot.
//!
//! Float64 has no wider float to carry its sums in, so each operand is cut in
//! two. Every value of a row of the left operand lies below `2^e` for that
//! row's `e`, and `(v + σ) - σ` against `σ = 3 · 2^(e+52-β)` leaves its first
//! piece -- the whole multiple of `2^(e+1-β)` nearest it, of at most `2^e` --
//! and, taken from the value, its rest, of at most `2^(e-β)`. The columns of
//! the right operand are cut alike, below `2^f`. A dot is taken in runs of at
//! most [`RUN`] terms, and `β` is the widest that keeps a run of the first
//! pieces' products -- whole multiples of `2^(e+f+2-2β)` -- and every partial
//! sum of them, in whatever order a kernel adds them, within `2^53` of those
//! multiples, where float64 holds them. So of the three matrix products each
//! run of each entry is made of,
//!
//! - the firsts by the firsts is exact, and
//! - the rests by the firsts and the values by the rests, `2^-β` of the
//!   first in size, are each off their exact value by at most `γ_w` of the
//!   sum of their terms' magnitudes for a run of `w`, whatever order it was
//!   added in; over a whole dot those sums are within the row's and the
//!   column's 2-norms, pieces by pieces.
//!
//! The runs' products are added into a float64 pair without error. Where
//! the bound narrows the pair to one float64, that is the entry; where it
//! does not -- one entry in a thousand of ordinary data sixty deep, one in
//! two hundred a thousand deep, and every one that cancels far below its
//! terms -- the entry is the exact dot of its row and column. A row and a
//! column of whole multiples of `2^(e+1-β)` and `2^(f+1-β)` have no rests,
//! so their pair is their entry: whole numbers, one-hot rows and zeros among
//! them, a zero a positive one, as a dot's is.
//!
//! The output is taken a band of [`ROWS`] rows by [`COLUMNS`] columns at a
//! time, so that a band's pair and the run being added to it stay in cache
//! however many runs pass through them. A row or column holding a value that
//! is not finite, or whose largest magnitude is past `2^399` or below
//! `2^-401`, is left to the exact dots, whose rules for those are the rule
//! everywhere a dot is taken; so is a product with a single row or column,
//! or too few entries to pay for the cutting.

use crate::error::{MinitensorError, Result};
use crate::ops::arithmetic::exact_sum::{Output, dot_f64, exact_dot_to, exact_quads_f64};
use crate::ops::arithmetic::rounded_once::{product_error, two_sum};
use crate::ops::map::{par_out_chunks, par_out_chunks_mapped, par_out_chunks_sized};
use crate::ops::provider::Storage;
use crate::tensor::{DataType, TensorData};

use super::Affine;

/// Float64's unit roundoff, `2^-53`.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// The most terms of a dot one run takes.
const RUN: usize = 64;

/// The most rows of the output a band takes, and the most columns.
const ROWS: usize = 128;
const COLUMNS: usize = 256;

/// Elements a task of the passes over an operand takes, at least.
const TASK: usize = 1 << 14;

/// Multiplications below which a product is taken on the calling thread:
/// some half a millisecond of it, under which the pool's round trips cost
/// more than they spread.
const SERIAL_BELOW: usize = 1 << 21;

/// Columns of the right operand cut at a time where they lie across its
/// storage.
const TILE: usize = 32;

/// What an entry the bound leaves open is marked with until its exact dot
/// is taken: a NaN, and one of another payload where its row and column
/// were cut and its pair came to nothing -- as the entries of a product of
/// two triangular matrices do outside their triangle, every term of theirs
/// being a zero.
const OPEN: f64 = f64::NAN;
const OPEN_AT_ZERO: f64 = f64::from_bits(0x7ff8_0000_0000_0001);

/// A function compiled for the target's baseline and, on x86_64, again for
/// AVX2, which is taken where the CPU has it: the passes either side of the
/// kernel's products are plain loops over a band, which twice the lanes
/// take in half the time. What the function calls is inlined, so that it is
/// compiled with the feature too.
macro_rules! wide {
    ($(#[$doc:meta])* fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $out:ty)? = $with:ident) => {
        $(#[$doc])*
        fn $name($($arg: $ty),*) $(-> $out)? {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2")]
                fn wide($($arg: $ty),*) $(-> $out)? {
                    $with($($arg),*)
                }
                // SAFETY: the feature was detected on this CPU just above.
                return unsafe { wide($($arg),*) };
            }
            $with($($arg),*)
        }
    };
}

/// A float dtype whose matrix products are rounded once, for what is
/// written once for both.
trait Rounded: Copy {
    /// The float of this width that `value` is.
    fn of(value: f64) -> Self;
    fn slice(data: &TensorData) -> Option<&[Self]>;
    fn slice_mut(data: &mut TensorData) -> Option<&mut [Self]>;
    /// `batch` products of `(m, k)` by `(k, n)`, each entry taken to
    /// `alpha · dot + beta · c` by `affine` and rounded once.
    #[allow(clippy::too_many_arguments)]
    fn affine_products(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        lhs: (&[Self], Storage),
        rhs: (&[Self], Storage),
        affine: Option<Affine<'_, Self>>,
        out: &mut [Self],
    ) -> Result<()>;
}

macro_rules! rounded {
    ($ty:ty, $slice:ident, $slice_mut:ident, $products:path) => {
        impl Rounded for $ty {
            fn of(value: f64) -> Self {
                value as $ty
            }

            fn slice(data: &TensorData) -> Option<&[Self]> {
                data.$slice()
            }

            fn slice_mut(data: &mut TensorData) -> Option<&mut [Self]> {
                data.$slice_mut()
            }

            fn affine_products(
                batch: usize,
                m: usize,
                k: usize,
                n: usize,
                lhs: (&[Self], Storage),
                rhs: (&[Self], Storage),
                affine: Option<Affine<'_, Self>>,
                out: &mut [Self],
            ) -> Result<()> {
                $products(batch, m, k, n, lhs, rhs, affine, out)
            }
        }
    };
}

rounded!(
    f32,
    as_f32_slice,
    as_f32_slice_mut,
    super::rounded_affine_products_f32
);
rounded!(
    f64,
    as_f64_slice,
    as_f64_slice_mut,
    rounded_affine_products_f64
);

/// What a product's entries are taken to before their one rounding,
/// whatever their width: `alpha · dot + beta · c`, the scales floats of
/// that width, with `c` the addend at `strides[0] · b + strides[1] · i +
/// strides[2] · j` of `addend`'s values, or zero where there is none.
#[derive(Clone, Copy)]
pub(crate) struct Taken<'a> {
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
    pub(crate) addend: Option<(&'a TensorData, [usize; 3])>,
}

/// `out[b] = lhs[b] @ rhs[b]` for each of `batch` products of `(m, k)` by
/// `(k, n)` of float32 or of float64 operands, either read in the layout it
/// is stored in, every entry the float nearest its exact value: its dot, or
/// what `taken` takes that to. `out` need not arrive zeroed.
pub(crate) fn rounded_products(
    dims: (usize, usize, usize, usize),
    lhs: (&TensorData, Storage),
    rhs: (&TensorData, Storage),
    taken: Option<Taken<'_>>,
    out: &mut TensorData,
) -> Result<()> {
    fn of<T: Rounded>(
        (batch, m, k, n): (usize, usize, usize, usize),
        lhs: (&TensorData, Storage),
        rhs: (&TensorData, Storage),
        taken: Option<Taken<'_>>,
        out: &mut TensorData,
    ) -> Result<()> {
        let missing = || MinitensorError::internal_error("a product's operands of one float");
        let zero = [T::of(0.0)];
        let affine = match taken {
            None => None,
            Some(taken) => Some(Affine {
                alpha: T::of(taken.alpha),
                beta: T::of(taken.beta),
                values: match taken.addend {
                    Some((data, _)) => T::slice(data).ok_or_else(missing)?,
                    None => &zero[..],
                },
                strides: taken.addend.map_or([0; 3], |(_, strides)| strides),
            }),
        };
        T::affine_products(
            batch,
            m,
            k,
            n,
            (T::slice(lhs.0).ok_or_else(missing)?, lhs.1),
            (T::slice(rhs.0).ok_or_else(missing)?, rhs.1),
            affine,
            T::slice_mut(out).ok_or_else(missing)?,
        )
    }
    match out.dtype() {
        DataType::Float32 => of::<f32>(dims, lhs, rhs, taken, out),
        DataType::Float64 => of::<f64>(dims, lhs, rhs, taken, out),
        _ => Err(MinitensorError::invalid_operation(
            "a rounded product takes float operands",
        )),
    }
}

/// `alpha`, `beta` and the addend of entry `(i, j)` of one product.
type Scaling<'a> = (f64, f64, &'a (dyn Fn(usize, usize) -> f64 + Sync));

/// `out[b] = lhs[b] @ rhs[b]` for each of `batch` products of `(m, k)` by
/// `(k, n)`, either operand read in the layout it is stored in, every entry
/// the float64 nearest its exact dot. `out` need not arrive zeroed.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn rounded_products_f64(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f64],
    lhs_storage: Storage,
    rhs: &[f64],
    rhs_storage: Storage,
    out: &mut [f64],
) -> Result<()> {
    rounded_affine_products_f64(
        batch,
        m,
        k,
        n,
        (lhs, lhs_storage),
        (rhs, rhs_storage),
        None,
        out,
    )
}

/// `out[b] = lhs[b] @ rhs[b]` for each of `batch` products of `(m, k)` by
/// `(k, n)`, either operand read in the layout it is stored in, every entry
/// the float64 nearest its exact dot -- or, taken to `alpha · dot + beta ·
/// c` by `affine`, where there is one, the float64 nearest that exactly.
/// `out` need not arrive zeroed.
///
/// The pair is scaled and the addend added to it with the exact errors of
/// both products kept, off the exact value by `|alpha|` times the dot's bound
/// and the roundings of the errors' sum. An entry left open is the exact sum
/// of `alpha · a_l · b_l` and `beta · c`.
#[allow(clippy::too_many_arguments)]
fn rounded_affine_products_f64(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    (lhs, lhs_storage): (&[f64], Storage),
    (rhs, rhs_storage): (&[f64], Storage),
    affine: Option<Affine<'_, f64>>,
    out: &mut [f64],
) -> Result<()> {
    let addend_reach = affine.map_or(0, |affine| {
        let [b, i, j] = affine.strides;
        b * batch.saturating_sub(1) + i * m.saturating_sub(1) + j * n.saturating_sub(1)
    });
    if affine.is_some_and(|affine| !out.is_empty() && addend_reach >= affine.values.len()) {
        return Err(MinitensorError::internal_error(format!(
            "a product's addend holds {} elements where its strides reach {addend_reach}",
            affine.map_or(0, |affine| affine.values.len())
        )));
    }
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
    let addend = |product: usize, i: usize, j: usize| {
        affine.map_or(0.0, |affine| {
            let [b, row, column] = affine.strides;
            affine.values[b * product + row * i + column * j]
        })
    };
    if k == 0 {
        // An empty dot is a zero, so each entry is `alpha · 0 + beta · c`.
        for (index, slot) in out.iter_mut().enumerate() {
            let (product, i, j) = (index / (m * n), index / n % m, index % n);
            *slot = match affine {
                None => 0.0,
                Some(affine) => exact_dot_to::<f64>(
                    [(affine.alpha, 0.0), (affine.beta, addend(product, i, j))].into_iter(),
                ),
            };
        }
        return Ok(());
    }
    let both_across = lhs_storage == Storage::Transposed && rhs_storage == Storage::Transposed;
    if (m == 1 || n == 1) && affine.is_none() && !both_across {
        // A set of dots, which the kernels take one by one as `dot` does.
        out.fill(0.0);
        return super::stored_gemm_f64(batch, m, k, n, lhs, lhs_storage, rhs, rhs_storage, out);
    }
    let one = |product: usize, out: &mut [f64]| -> Result<()> {
        let left = Side::of(
            &lhs[product * m * k..(product + 1) * m * k],
            lhs_storage == Storage::RowMajor,
            m,
            k,
        );
        let right = Side::of(
            &rhs[product * k * n..(product + 1) * k * n],
            rhs_storage == Storage::Transposed,
            n,
            k,
        );
        let entry_addend = |i: usize, j: usize| addend(product, i, j);
        let scaling = affine.map(|affine| (affine.alpha, affine.beta, &entry_addend as _));
        product_f64(m, k, n, left, right, scaling, out);
        Ok(())
    };
    if batch > 1 && batch >= crate::parallel::current_num_threads() {
        crate::ops::map::try_par_out_chunks(out, m * n, &|start, part| one(start / (m * n), part))
    } else {
        out.chunks_mut(m * n)
            .enumerate()
            .try_for_each(|(product, part)| one(product, part))
    }
}

/// One operand, read by the index it keeps -- a row of the left operand, a
/// column of the right -- and the index its dots run along.
#[derive(Clone, Copy)]
struct Side<'a> {
    values: &'a [f64],
    /// How many rows, or columns, it has.
    count: usize,
    /// Steps through `values` from one row or column to the next, and along
    /// one.
    across: usize,
    along: usize,
}

impl<'a> Side<'a> {
    /// `count` runs of `k`, each laid out along its storage where `upright`,
    /// and across it otherwise.
    fn of(values: &'a [f64], upright: bool, count: usize, k: usize) -> Self {
        let (across, along) = if upright { (k, 1) } else { (1, count) };
        Side {
            values,
            count,
            across,
            along,
        }
    }

    /// Whether each row or column lies along the storage, its terms side by
    /// side.
    fn upright(&self) -> bool {
        self.along == 1
    }

    #[inline(always)]
    fn at(&self, index: usize, l: usize) -> f64 {
        self.values[index * self.across + l * self.along]
    }
}

/// `2^power`, a normal float64.
fn power(power: i32) -> f64 {
    f64::from_bits(((power + 1023) as u64) << 52)
}

/// The largest magnitude in `source`, and bits that are zero, but for the
/// sign's, where every value is finite: a finite value times zero is a
/// zero, and any other a NaN.
#[inline(always)]
fn peak_line_with(source: &[f64]) -> (f64, u64) {
    let (mut peaks, mut marks) = ([0.0f64; 4], [0u64; 4]);
    let (chunks, tail) = source.as_chunks::<4>();
    for chunk in chunks {
        for lane in 0..4 {
            let magnitude = chunk[lane].abs();
            peaks[lane] = if magnitude > peaks[lane] {
                magnitude
            } else {
                peaks[lane]
            };
            marks[lane] |= (chunk[lane] * 0.0).to_bits();
        }
    }
    for &value in tail {
        peaks[0] = if value.abs() > peaks[0] {
            value.abs()
        } else {
            peaks[0]
        };
        marks[0] |= (value * 0.0).to_bits();
    }
    (
        peaks.into_iter().fold(0.0, f64::max),
        marks.into_iter().fold(0, |all, mark| all | mark),
    )
}

wide! {
    /// [`peak_line_with`], in the widest lanes the CPU has.
    fn peak_line(source: &[f64]) -> (f64, u64) = peak_line_with
}

/// [`peak_line_with`] of each of several rows or columns, one value of
/// each side by side in `source`.
#[inline(always)]
fn peak_side_with(source: &[f64], peaks: &mut [f64], marks: &mut [u64]) {
    for ((&value, peak), mark) in source.iter().zip(peaks).zip(marks) {
        let magnitude = value.abs();
        *peak = if magnitude > *peak { magnitude } else { *peak };
        *mark |= (value * 0.0).to_bits();
    }
}

wide! {
    /// [`peak_side_with`], in the widest lanes the CPU has.
    fn peak_side(source: &[f64], peaks: &mut [f64], marks: &mut [u64]) = peak_side_with
}

/// How the values of each row of the left operand, or column of the right,
/// are cut.
struct Cuts {
    /// `3 · 2^(e+52-β)`, which leaves the first piece.
    first: Vec<f64>,
    /// Every bit, or none for a row or column left to the exact dots, whose
    /// values are cut as zeros.
    keep: Vec<u64>,
    /// `2^e`, which the bound is scaled by: zero for a row of zeros, and NaN
    /// for one left to the exact dots, which settles nothing.
    scale: Vec<f64>,
    /// `2^-e`, and zero where the scale is zero or NaN.
    shrink: Vec<f64>,
}

/// What cuts one row's or column's values, and scales them and their rests
/// for their squares: its `first`, its `keep`, its `shrink` and `2^β` of
/// that.
type Piece = (f64, u64, f64, f64);

impl Cuts {
    /// The cut of each row, or column, of `side` into a first piece of
    /// `width` bits and a rest, from the largest magnitude along it.
    fn of(side: Side<'_>, k: usize, width: i32) -> Cuts {
        let mut peaks = vec![0.0f64; side.count];
        let mut marks = vec![0u64; side.count];
        let bytes = side.count * k * 8;
        if side.upright() {
            let mut both: Vec<(f64, u64)> = vec![(0.0, 0); side.count];
            par_out_chunks_sized(&mut both, 64, bytes, &|first, part| {
                for (offset, slot) in part.iter_mut().enumerate() {
                    *slot = peak_line(&side.values[(first + offset) * side.across..][..k]);
                }
            });
            for ((peak, mark), both) in peaks.iter_mut().zip(marks.iter_mut()).zip(both) {
                (*peak, *mark) = both;
            }
        } else {
            for l in 0..k {
                let source = &side.values[l * side.along..][..side.count];
                peak_side(source, &mut peaks, &mut marks);
            }
        }
        let mut cuts = Cuts {
            first: Vec::with_capacity(side.count),
            keep: Vec::with_capacity(side.count),
            scale: Vec::with_capacity(side.count),
            shrink: Vec::with_capacity(side.count),
        };
        for (peak, mark) in peaks.into_iter().zip(marks) {
            // The row lies below `2^e` for its largest magnitude's exponent
            // field less 1022.
            let (finite, e) = (mark << 1 == 0, (peak.to_bits() >> 52) as i32 - 1022);
            let (first, keep, scale, shrink) = if peak == 0.0 && finite {
                (1.0, u64::MAX, 0.0, 0.0)
            } else if finite && (-400..=400).contains(&e) {
                (3.0 * power(e + 52 - width), u64::MAX, power(e), power(-e))
            } else {
                (1.0, 0, f64::NAN, 0.0)
            };
            cuts.first.push(first);
            cuts.keep.push(keep);
            cuts.scale.push(scale);
            cuts.shrink.push(shrink);
        }
        cuts
    }

    fn piece(&self, index: usize, grow: f64) -> Piece {
        (
            self.first[index],
            self.keep[index],
            self.shrink[index],
            self.shrink[index] * grow,
        )
    }
}

/// What the values of each of several rows or columns, cut, sum to: the
/// squares of the values over `2^e` and of their rests over `2^(e-β)`, and
/// every bit any rest has.
struct Sums {
    values: Vec<f64>,
    rests: Vec<f64>,
    bits: Vec<u64>,
}

impl Sums {
    fn of(count: usize) -> Sums {
        Sums {
            values: vec![0.0; count],
            rests: vec![0.0; count],
            bits: vec![0; count],
        }
    }

    fn add(&mut self, index: usize, (values, rests, bits): (f64, f64, u64)) {
        self.values[index] += values;
        self.rests[index] += rests;
        self.bits[index] |= bits;
    }

    /// The [`Norms`] of row or column `index` of `k` values, cut `width`
    /// bits down from `scale`.
    ///
    /// A square below the normals was added as less than it is, by under
    /// `2^-1021` of the scaled sum each: nothing beside the values' sum,
    /// which holds a square of a quarter at least, and at most `√k · 2^-510`
    /// of the rests' norm, which is added to it.
    fn norms(&self, index: usize, scale: f64, k: usize, width: i32) -> Norms {
        let whole = self.bits[index] << 1 == 0 && !scale.is_nan();
        let slack = (k as f64).sqrt() * 2f64.powi(-510);
        Norms {
            values: scale * self.values[index].sqrt(),
            rests: if whole {
                0.0
            } else {
                scale * power(-width) * (self.rests[index].sqrt() + slack)
            },
            whole,
        }
    }
}

/// What a row or column brings to its entries' bounds: no less than the
/// 2-norms of its values and of their rests, each short of the roundings
/// the bound's own factor allows for, and whether it has no rests.
#[derive(Clone, Copy)]
struct Norms {
    values: f64,
    rests: f64,
    whole: bool,
}

/// The values of one row or column cut into their first pieces and their
/// rests -- a value of one left to the exact dots cut as zero -- and what
/// they add to its [`Sums`].
#[inline(always)]
fn cut_line_with(
    source: &[f64],
    (first, keep, shrink, grow): Piece,
    ones: &mut [f64],
    rests: &mut [f64],
) -> (f64, f64, u64) {
    let (mut values, mut squares, mut bits) = ([0.0f64; 4], [0.0f64; 4], [0u64; 4]);
    let mut take = |lane: usize, value: f64, one: &mut f64, rest: &mut f64| {
        let value = f64::from_bits(value.to_bits() & keep);
        *one = (value + first) - first;
        *rest = value - *one;
        let (scaled, scaled_rest) = (value * shrink, *rest * grow);
        values[lane] += scaled * scaled;
        squares[lane] += scaled_rest * scaled_rest;
        bits[lane] |= rest.to_bits();
    };
    let (chunks, tail) = source.as_chunks::<4>();
    let (one_chunks, one_tail) = ones.as_chunks_mut::<4>();
    let (rest_chunks, rest_tail) = rests.as_chunks_mut::<4>();
    for ((chunk, ones), rests) in chunks.iter().zip(one_chunks).zip(rest_chunks) {
        for lane in 0..4 {
            take(lane, chunk[lane], &mut ones[lane], &mut rests[lane]);
        }
    }
    for ((&value, one), rest) in tail.iter().zip(one_tail).zip(rest_tail) {
        take(0, value, one, rest);
    }
    (
        values.iter().sum(),
        squares.iter().sum(),
        bits.into_iter().fold(0, |all, bits| all | bits),
    )
}

wide! {
    /// [`cut_line_with`], in the widest lanes the CPU has.
    fn cut_line(source: &[f64], piece: Piece, ones: &mut [f64], rests: &mut [f64]) -> (f64, f64, u64) = cut_line_with
}

/// One value of each of several rows or columns, side by side in `source`,
/// cut as [`cut_line_with`] cuts them, each by its own `cuts` from `from`,
/// with what it adds to its `sums`.
#[inline(always)]
fn cut_side_with(
    source: &[f64],
    (cuts, from, grow): (&Cuts, usize, f64),
    (values, squares, bits): (&mut [f64], &mut [f64], &mut [u64]),
    ones: &mut [f64],
    rests: &mut [f64],
) {
    let count = source.len();
    let (firsts, keeps, shrinks) = (
        &cuts.first[from..][..count],
        &cuts.keep[from..][..count],
        &cuts.shrink[from..][..count],
    );
    let (values, squares, bits) = (
        &mut values[..count],
        &mut squares[..count],
        &mut bits[..count],
    );
    let (ones, rests) = (&mut ones[..count], &mut rests[..count]);
    for index in 0..count {
        let value = f64::from_bits(source[index].to_bits() & keeps[index]);
        let one = (value + firsts[index]) - firsts[index];
        let rest = value - one;
        let (scaled, scaled_rest) = (value * shrinks[index], rest * (shrinks[index] * grow));
        values[index] += scaled * scaled;
        squares[index] += scaled_rest * scaled_rest;
        bits[index] |= rest.to_bits();
        ones[index] = one;
        rests[index] = rest;
    }
}

/// A tile of columns of the right operand, lying across its storage, cut
/// for one run of `span` rows into `columns`: each column's rests and then
/// its first pieces. `rows` is the operand from the run's first row and the
/// tile's first column, its rows `along` apart.
#[inline(always)]
fn cut_tile_with(
    (rows, along, span): (&[f64], usize, usize),
    cuts: (&Cuts, usize, f64),
    (values, squares, bits): (&mut [f64], &mut [f64], &mut [u64]),
    columns: &mut [f64],
) {
    let count = columns.len() / (2 * span);
    let (mut ones, mut rests) = ([0.0; TILE], [0.0; TILE]);
    for step in 0..span {
        cut_side_with(
            &rows[step * along..][..count],
            cuts,
            (&mut *values, &mut *squares, &mut *bits),
            &mut ones,
            &mut rests,
        );
        for (j, column) in columns.chunks_exact_mut(2 * span).enumerate() {
            column[step] = rests[j];
            column[span + step] = ones[j];
        }
    }
}

wide! {
    /// [`cut_tile_with`], in the widest lanes the CPU has.
    fn cut_tile(source: (&[f64], usize, usize), cuts: (&Cuts, usize, f64), sums: (&mut [f64], &mut [f64], &mut [u64]), columns: &mut [f64]) = cut_tile_with
}

/// A band of rows of the left operand, lying across its storage, cut into
/// `ones` and `rests`, `k` values a row. `columns` is the operand from the
/// band's first row, its columns `along` apart.
#[inline(always)]
fn cut_band_with(
    (columns, along, k): (&[f64], usize, usize),
    cuts: (&Cuts, usize, f64),
    (values, squares, bits): (&mut [f64], &mut [f64], &mut [u64]),
    (ones, rests): (&mut [f64], &mut [f64]),
) {
    let rows = ones.len() / k;
    let (mut one, mut rest) = ([0.0; ROWS], [0.0; ROWS]);
    for l in 0..k {
        cut_side_with(
            &columns[l * along..][..rows],
            cuts,
            (&mut *values, &mut *squares, &mut *bits),
            &mut one,
            &mut rest,
        );
        for row in 0..rows {
            ones[row * k + l] = one[row];
            rests[row * k + l] = rest[row];
        }
    }
}

wide! {
    /// [`cut_band_with`], in the widest lanes the CPU has.
    fn cut_band(source: (&[f64], usize, usize), cuts: (&Cuts, usize, f64), sums: (&mut [f64], &mut [f64], &mut [u64]), planes: (&mut [f64], &mut [f64])) = cut_band_with
}

/// The right operand cut into `out`, run after run: for each run, each
/// column's rests and then its first pieces, column after column, so that
/// a column's part of a run lies side by side; and each column's [`Sums`].
fn cut_right(
    side: Side<'_>,
    cuts: &Cuts,
    (run, width, spread): (usize, i32, bool),
    out: &mut [f64],
) -> Sums {
    let n = side.count;
    let grow = power(width);
    // Whole runs to a task, each task summing its part of every column.
    let task = if spread {
        (TASK / (2 * run * n)).max(1) * 2 * run * n
    } else {
        out.len()
    };
    let parts = par_out_chunks_mapped(out, task, &|start, part| {
        let mut sums = Sums::of(n);
        let mut first = start / (2 * n);
        for block in part.chunks_mut(2 * run * n) {
            let span = block.len() / (2 * n);
            if side.upright() {
                for (j, column) in block.chunks_exact_mut(2 * span).enumerate() {
                    let source = &side.values[j * side.across + first..][..span];
                    let (rests, ones) = column.split_at_mut(span);
                    sums.add(j, cut_line(source, cuts.piece(j, grow), ones, rests));
                }
            } else {
                // A tile of columns at a time, so that the lines a run's
                // rows are written down stay in cache from row to row.
                for (tile, columns) in block.chunks_mut(TILE * 2 * span).enumerate() {
                    let from = tile * TILE;
                    cut_tile(
                        (&side.values[first * side.along + from..], side.along, span),
                        (cuts, from, grow),
                        (
                            &mut sums.values[from..],
                            &mut sums.rests[from..],
                            &mut sums.bits[from..],
                        ),
                        columns,
                    );
                }
            }
            first += span;
        }
        sums
    });
    let mut sums = Sums::of(n);
    for part in parts {
        for j in 0..n {
            sums.add(j, (part.values[j], part.rests[j], part.bits[j]));
        }
    }
    sums
}

/// Rows `first_row..` of the left operand cut into `out`: their first
/// pieces, `k` a row, and then their rests; and each row's [`Sums`].
fn cut_left(
    side: Side<'_>,
    cuts: &Cuts,
    first_row: usize,
    k: usize,
    width: i32,
    out: &mut [f64],
) -> Sums {
    let rows = out.len() / (2 * k);
    let grow = power(width);
    let mut sums = Sums::of(rows);
    let (ones, rests) = out.split_at_mut(rows * k);
    if side.upright() {
        for (row, (ones, rests)) in ones
            .chunks_exact_mut(k)
            .zip(rests.chunks_exact_mut(k))
            .enumerate()
        {
            let i = first_row + row;
            let source = &side.values[i * side.across..][..k];
            sums.add(row, cut_line(source, cuts.piece(i, grow), ones, rests));
        }
    } else {
        cut_band(
            (&side.values[first_row..], side.along, k),
            (cuts, first_row, grow),
            (&mut sums.values, &mut sums.rests, &mut sums.bits),
            (ones, rests),
        );
    }
    sums
}

/// `(high, low) += terms` as a float64 pair: each term added by a two-sum,
/// its error into the low part.
#[inline(always)]
fn add_exact_with(high: &mut [f64], low: &mut [f64], terms: &[f64]) {
    for ((high, low), &term) in high.iter_mut().zip(low.iter_mut()).zip(terms) {
        let (sum, error) = two_sum(*high, term);
        *high = sum;
        *low += error;
    }
}

wide! {
    /// [`add_exact_with`], in the widest lanes the CPU has.
    fn add_exact(high: &mut [f64], low: &mut [f64], terms: &[f64]) = add_exact_with
}

/// `(high, low) += first + second`, the sum of the two rounded and then
/// added as [`add_exact_with`] adds a term.
#[inline(always)]
fn add_sum_with(high: &mut [f64], low: &mut [f64], first: &[f64], second: &[f64]) {
    for (((high, low), &first), &second) in
        high.iter_mut().zip(low.iter_mut()).zip(first).zip(second)
    {
        let (sum, error) = two_sum(*high, first + second);
        *high = sum;
        *low += error;
    }
}

wide! {
    /// [`add_sum_with`], in the widest lanes the CPU has.
    fn add_sum(high: &mut [f64], low: &mut [f64], first: &[f64], second: &[f64]) = add_sum_with
}

/// A row of pairs narrowed into `line`, each to the float64 its bound
/// settles it as or marked [`OPEN`] where it settles none, and how many
/// were left so: the bound is `by.0` of the column's rests' norm and `by.1`
/// of its two norms' sum, and nothing where the row and the column are both
/// whole.
#[inline(always)]
fn settle_line_with(
    line: &mut [f64],
    (highs, lows): (&[f64], &[f64]),
    (rests, sums, wholes): (&[f64], &[f64], &[bool]),
    by: (f64, f64, bool),
) -> usize {
    let count = line.len();
    let (highs, lows) = (&highs[..count], &lows[..count]);
    let (rests, sums, wholes) = (&rests[..count], &sums[..count], &wholes[..count]);
    let mut open = 0;
    for index in 0..count {
        let bound = if by.2 & wholes[index] {
            0.0
        } else {
            by.0 * rests[index] + by.1 * sums[index]
        };
        let (value, rest) = two_sum(highs[index], lows[index]);
        let (value, settled) = f64::settle_lanes(value, rest, bound);
        let open_as = if (value == 0.0) & !bound.is_nan() {
            OPEN_AT_ZERO
        } else {
            OPEN
        };
        line[index] = if settled { value + 0.0 } else { open_as };
        open += usize::from(!settled);
    }
    open
}

wide! {
    /// [`settle_line_with`], in the widest lanes the CPU has.
    fn settle_line(line: &mut [f64], pair: (&[f64], &[f64]), norms: (&[f64], &[f64], &[bool]), by: (f64, f64, bool)) -> usize = settle_line_with
}

/// `alpha · (high + low ± bound) + beta · addend` as a pair and its bound:
/// both products split into the float64 nearest them and its error, the
/// errors and the low part's product added up, each rounding of that inside
/// `4.01u` of the parts' magnitudes, and a subnormal's worth for products
/// so small their errors were rounded.
#[inline(always)]
fn scaled(
    (high, low, bound): (f64, f64, f64),
    alpha: f64,
    beta: f64,
    addend: f64,
) -> (f64, f64, f64) {
    let product = alpha * high;
    let product_rest = product_error(alpha, high, product);
    let low = alpha * low;
    let added = beta * addend;
    let added_rest = product_error(beta, addend, added);
    let (sum, sum_rest) = two_sum(product, added);
    let rest = ((product_rest + low) + added_rest) + sum_rest;
    let size = product_rest.abs() + low.abs() + added_rest.abs() + sum_rest.abs();
    let (value, value_rest) = two_sum(sum, rest);
    let bound =
        (alpha.abs() * bound + size * 4.01 * UNIT) * (1.0 + 2f64.powi(-40)) + f64::from_bits(16);
    (value, value_rest, bound)
}

/// What `terms`, each a product of three factors, make added one by one
/// where a factor is not finite -- NaN for a NaN, an infinity times a zero
/// or infinite products of both signs, and otherwise the infinity of their
/// sign -- or `None` where every factor is finite.
fn special(terms: impl Iterator<Item = [f64; 3]>) -> Option<f64> {
    let (mut nan, mut positive, mut negative) = (false, false, false);
    for factors in terms {
        if factors.iter().all(|factor| factor.is_finite()) {
            continue;
        }
        if factors
            .iter()
            .any(|factor| factor.is_nan() || *factor == 0.0)
        {
            nan = true;
        } else if factors
            .iter()
            .filter(|factor| factor.is_sign_negative())
            .count()
            % 2
            == 1
        {
            negative = true;
        } else {
            positive = true;
        }
    }
    if nan || positive && negative {
        Some(f64::NAN)
    } else if positive {
        Some(f64::INFINITY)
    } else if negative {
        Some(f64::NEG_INFINITY)
    } else {
        None
    }
}

/// The float64 nearest `alpha · Σ row[l] · column[l] + beta · addend`, or
/// what the products make added one by one where a value is not finite.
///
/// Each `alpha · row[l]` is split into the float64 nearest it and that
/// rounding's error, and the halves taken as `dot` takes its pairs; a split
/// that underflowed leaves the whole to the exact sum of the triple
/// products.
fn affine_dot(row: &[f64], column: &[f64], alpha: f64, beta: f64, addend: f64) -> f64 {
    let triples = || {
        row.iter()
            .zip(column)
            .map(|(&a, &b)| [alpha, a, b])
            .chain(std::iter::once([beta, addend, 1.0]))
    };
    if let Some(special) = special(triples()) {
        return special;
    }
    let mut left = Vec::with_capacity(2 * row.len() + 1);
    let mut right = Vec::with_capacity(2 * row.len() + 1);
    for (&a, &b) in row.iter().zip(column) {
        let product = alpha * a;
        let rest = product_error(alpha, a, product);
        let exact = if product == 0.0 {
            alpha == 0.0 || a == 0.0
        } else {
            product.abs() >= 2f64.powi(-960) && product.is_finite() && rest.is_finite()
        };
        if !exact {
            let terms: Vec<[f64; 4]> = triples().map(|[x, y, z]| [x, y, z, 1.0]).collect();
            return exact_quads_f64(&terms);
        }
        left.extend([product, rest]);
        right.extend([b, b]);
    }
    left.push(beta);
    right.push(addend);
    dot_f64(&left, &right)
}

/// Whether a product is its exact dots throughout: a single row or column,
/// or so few entries that their dots cost less than cutting the operands
/// would.
fn few(m: usize, k: usize, n: usize) -> bool {
    m == 1 || n == 1 || m * n * (250 + k + k / 4) < 12_000 + 10 * (m + n) * k
}

/// One product of `(m, k)` by `(k, n)` into `out`, each entry the float64
/// nearest its exact value.
fn product_f64(
    m: usize,
    k: usize,
    n: usize,
    left: Side<'_>,
    right: Side<'_>,
    scaling: Option<Scaling<'_>>,
    out: &mut [f64],
) {
    if few(m, k, n) {
        out.fill(OPEN);
        exact_entries(k, n, left, right, None, scaling, out, m * n);
    } else {
        split_product(m, k, n, left, right, scaling, out);
    }
}

/// The entries the pieces' products settle, and the exact dots of those
/// they leave open.
fn split_product(
    m: usize,
    k: usize,
    n: usize,
    left: Side<'_>,
    right: Side<'_>,
    scaling: Option<Scaling<'_>>,
    out: &mut [f64],
) {
    // Runs of one length, but for the last, and the widest first pieces
    // whose products a run of them sums exactly: at most `2^(β-1)` of their
    // grain each, so `run · 2^(2β-2)` of the products' at most.
    let run = k.div_ceil(k.div_ceil(RUN));
    let bits = (55 - (usize::BITS - (run - 1).leading_zeros()) as i32) / 2;
    let (left_cuts, right_cuts) = (Cuts::of(left, k, bits), Cuts::of(right, k, bits));
    let mut rights = vec![0.0; 2 * k * n];
    let spread = m * k * n >= SERIAL_BELOW;
    let sums = cut_right(right, &right_cuts, (run, bits, spread), &mut rights);
    // Each column's rests' norm, its two norms' sum, and whether it is
    // whole.
    let (mut right_rests, mut right_sums, mut right_wholes) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (j, &scale) in right_cuts.scale.iter().enumerate() {
        let norms = sums.norms(j, scale, k, bits);
        right_rests.push(norms.rests);
        right_sums.push(norms.values + norms.rests);
        right_wholes.push(norms.whole);
    }
    // The norms were summed and rooted within `(k + 4)u` of themselves,
    // and the bound is worked out from them within a few more. Its last
    // factor also covers the second products' terms rounded below the
    // normals, by `2^-1075` each: rows and columns that are cut lie above
    // `2^-401`, so a bound that is not zero is `λ · 2^-804` at least.
    let slack = (1.0 + 4.0 * (k as f64 + 8.0) * UNIT) * (1.0 + 2f64.powi(-30));
    // `γ_w` of each of a run's two rounded products and `u` of their sum,
    // within `γ_(w+2)` of their terms' magnitudes; and the roundings of the
    // pairs' low parts: each of the `2R` errors added to one is within `u`
    // of a high part, itself within `3.1` of the norms' product, and their
    // sum is within `γ_2R` of their magnitudes'.
    let terms = run as f64 + 2.0;
    let gamma = terms * UNIT / (1.0 - terms * UNIT) * slack;
    let folds = 2.0 * k.div_ceil(run) as f64 * UNIT;
    let lambda = 3.2 * folds * folds / (1.0 - folds) * slack;
    // Bands of one height, but for the last: as many as keep each within
    // `ROWS`, and one for each thread where that leaves them tall enough to
    // give the kernel its due.
    let threads = if spread {
        crate::parallel::current_num_threads()
    } else {
        1
    };
    let bands = m.div_ceil(ROWS).max(threads.min(m / 32));
    let rows = m.div_ceil(bands);
    let columns = n.div_ceil(n.div_ceil(COLUMNS));
    let rights = &rights[..];
    let band = |start: usize, part: &mut [f64]| -> usize {
        let (first_row, rows) = (start / n, part.len() / n);
        // The band's first pieces and, after them, their rests.
        let mut lefts = vec![0.0; 2 * rows * k];
        let sums = cut_left(left, &left_cuts, first_row, k, bits, &mut lefts);
        // The pair, the two products of a run the kernel takes together,
        // and its third.
        let mut work = vec![0.0; 5 * rows * columns];
        let mut open = 0;
        for first_column in (0..n).step_by(columns) {
            let width = columns.min(n - first_column);
            let size = rows * width;
            let (pair, products) = work.split_at_mut(2 * rows * columns);
            let (high, low) = pair.split_at_mut(rows * columns);
            let (high, low) = (&mut high[..size], &mut low[..size]);
            let (stacked, third) = products.split_at_mut(2 * rows * columns);
            let (stacked, third) = (&mut stacked[..2 * size], &mut third[..size]);
            high.fill(0.0);
            low.fill(0.0);
            for start in (0..k).step_by(run) {
                let span = run.min(k - start);
                let column = 2 * start * n + first_column * 2 * span;
                // SAFETY: `span` values from `start` of each of the `2 ·
                // rows` rows of `k` the left pieces hold, and of each of
                // `rows` rows of the left operand from `first_row`, lie
                // inside them; `span` values from either half of each of
                // `width` columns' `2 · span` of this run lie inside the
                // right pieces; and the outputs hold `2 · rows · width` and
                // `rows · width` values.
                unsafe {
                    // The first pieces and the rests, by the first pieces.
                    super::gemm_strided_serial_f64(
                        2 * rows,
                        span,
                        width,
                        lefts.as_ptr().add(start),
                        k,
                        1,
                        rights.as_ptr().add(column + span),
                        1,
                        2 * span,
                        stacked.as_mut_ptr(),
                        width,
                    );
                    // The values by the rests.
                    super::gemm_strided_serial_f64(
                        rows,
                        span,
                        width,
                        left.values
                            .as_ptr()
                            .add(first_row * left.across + start * left.along),
                        left.across,
                        left.along,
                        rights.as_ptr().add(column),
                        1,
                        2 * span,
                        third.as_mut_ptr(),
                        width,
                    );
                }
                let (exact, rounded) = stacked.split_at(size);
                add_exact(high, low, exact);
                add_sum(high, low, rounded, third);
            }
            for (row, ((line, highs), lows)) in part
                .chunks_exact_mut(n)
                .zip(high.chunks_exact(width))
                .zip(low.chunks_exact(width))
                .enumerate()
            {
                let by = sums.norms(row, left_cuts.scale[first_row + row], k, bits);
                let by = (
                    gamma * by.values,
                    gamma * by.rests + lambda * (by.values + by.rests),
                    by.whole,
                );
                let line = &mut line[first_column..][..width];
                let norms = (
                    &right_rests[first_column..][..width],
                    &right_sums[first_column..][..width],
                    &right_wholes[first_column..][..width],
                );
                let Some((alpha, beta, addend)) = scaling else {
                    open += settle_line(line, (highs, lows), norms, by);
                    continue;
                };
                for (j, slot) in line.iter_mut().enumerate() {
                    let bound = if by.2 & norms.2[j] {
                        0.0
                    } else {
                        by.0 * norms.0[j] + by.1 * norms.1[j]
                    };
                    let (value, rest) = two_sum(highs[j], lows[j]);
                    let (value, rest, bound) = scaled(
                        (value, rest, bound),
                        alpha,
                        beta,
                        addend(first_row + row, first_column + j),
                    );
                    let (value, settled) = f64::settle_lanes(value, rest, bound);
                    *slot = if settled { value + 0.0 } else { OPEN };
                    open += usize::from(!settled);
                }
            }
        }
        open
    };
    let open: usize = if spread {
        par_out_chunks_mapped(out, rows * n, &band)
            .into_iter()
            .sum()
    } else {
        out.chunks_mut(rows * n)
            .enumerate()
            .map(|(nth, part)| band(nth * rows * n, part))
            .sum()
    };
    if open > 0 {
        let pieces = Some((rights, run, &right_cuts.keep[..]));
        exact_entries(k, n, left, right, pieces, scaling, out, open);
    }
}

/// The right operand's pieces as [`cut_right`] lays them, the length of
/// their runs, and each column's mask: none for one cut as zeros.
type Pieces<'a> = (&'a [f64], usize, &'a [u64]);

/// The first and the last place at which `values` holds other than a
/// zero, or `None` for nothing but zeros.
fn support(values: &[f64]) -> Option<(usize, usize)> {
    let first = values.iter().position(|&value| value != 0.0)?;
    let last = values.iter().rposition(|&value| value != 0.0)?;
    Some((first, last))
}

/// Whether two [`support`]s share no place, so that every product of the
/// finite values they are of is a zero exactly.
fn apart(a: Option<(usize, usize)>, b: Option<(usize, usize)>) -> bool {
    match (a, b) {
        (Some((a_first, a_last)), Some((b_first, b_last))) => a_last < b_first || b_last < a_first,
        _ => true,
    }
}

/// Whether every product of `a` and `b` is a zero exactly: one factor a
/// zero and the other finite, so that their dot is a zero with nothing
/// rounded on the way.
fn no_products(a: &[f64], b: &[f64]) -> bool {
    a.chunks(64).zip(b.chunks(64)).all(|(a, b)| {
        let mut nothing = true;
        for (&x, &y) in a.iter().zip(b) {
            nothing &= ((x == 0.0) | (y == 0.0)) & (x * y == 0.0);
        }
        nothing
    })
}

/// Each of the `open` entries of `out` left marked as the exact value of its
/// row and column: a zero where it was marked [`OPEN_AT_ZERO`] and every
/// one of its products is one -- which the places its row and its column
/// hold anything at say at a glance, where they lie apart, as a triangle's
/// do -- and `dot` of the two otherwise. A column lying across its storage
/// is put together from its `pieces`, each value the sum of its two, where
/// there are any.
#[allow(clippy::too_many_arguments)]
fn exact_entries(
    k: usize,
    n: usize,
    left: Side<'_>,
    right: Side<'_>,
    pieces: Option<Pieces<'_>>,
    scaling: Option<Scaling<'_>>,
    out: &mut [f64],
    open: usize,
) {
    // The right operand's columns are laid along their storage where they
    // lie across it, when enough entries are open to pay for laying them
    // all: more where the pieces hold each column's runs side by side.
    let lay = !right.upright() && open * 4 > if pieces.is_some() { 8 * n } else { n };
    let mut laid = Vec::new();
    if lay {
        laid = vec![0.0; n * k];
        let band = (TASK / k).max(1);
        par_out_chunks(&mut laid, band * k, &|first, part| {
            let (first, width) = (first / k, part.len() / k);
            for l in 0..k {
                let row = &right.values[l * right.along + first..][..width];
                for (column, &value) in row.iter().enumerate() {
                    part[column * k + l] = value;
                }
            }
        });
    }
    // Each column's support, found the first time an entry of it is open
    // at a zero and kept for the rest.
    type Support = std::sync::OnceLock<Option<(usize, usize)>>;
    let supports: Vec<Support> = (0..n).map(|_| Support::new()).collect();
    par_out_chunks_sized(out, n, open * k * 16, &|start, part| {
        let (mut row, mut column) = (Vec::new(), Vec::new());
        for (offset, line) in part.chunks_mut(n).enumerate() {
            if !line.iter().any(|value| value.is_nan()) {
                continue;
            }
            let i = start / n + offset;
            let row: &[f64] = if left.upright() {
                &left.values[i * k..][..k]
            } else {
                row.clear();
                row.extend((0..k).map(|l| left.at(i, l)));
                &row
            };
            let mut row_support = None;
            for (j, slot) in line.iter_mut().enumerate() {
                if !slot.is_nan() {
                    continue;
                }
                let at_zero = scaling.is_none() && slot.to_bits() == OPEN_AT_ZERO.to_bits();
                if at_zero
                    && let Some(&of_column) = supports[j].get()
                    && apart(*row_support.get_or_insert_with(|| support(row)), of_column)
                {
                    *slot = 0.0;
                    continue;
                }
                let column: &[f64] = if right.upright() {
                    &right.values[j * k..][..k]
                } else if lay {
                    &laid[j * k..][..k]
                } else {
                    column.clear();
                    match pieces {
                        Some((pieces, run, keep)) if keep[j] != 0 => {
                            for block in pieces.chunks(2 * run * n) {
                                let span = block.len() / (2 * n);
                                let (rests, ones) =
                                    block[j * 2 * span..][..2 * span].split_at(span);
                                column.extend(rests.iter().zip(ones).map(|(rest, one)| rest + one));
                            }
                        }
                        _ => column.extend((0..k).map(|l| right.at(j, l))),
                    }
                    &column
                };
                if at_zero {
                    let of_column = *supports[j].get_or_init(|| support(column));
                    let of_row = *row_support.get_or_insert_with(|| support(row));
                    if apart(of_row, of_column) || no_products(row, column) {
                        *slot = 0.0;
                        continue;
                    }
                }
                *slot = match scaling {
                    None => dot_f64(row, column),
                    Some((alpha, beta, addend)) => {
                        affine_dot(row, column, alpha, beta, addend(i, j))
                    }
                };
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(seed: u64, len: usize, spread: bool) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                if spread {
                    unit * 2f64.powi((state % 41) as i32 - 20)
                } else {
                    unit
                }
            })
            .collect()
    }

    fn at(storage: Storage, rows: usize, cols: usize, r: usize, c: usize) -> usize {
        match storage {
            Storage::RowMajor => r * cols + c,
            Storage::Transposed => c * rows + r,
        }
    }

    /// Each entry's exact value: the exact sum of its triple products, or,
    /// for a product too large to take that way in a test's time, `dot` of
    /// its row and column.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        batch: usize,
        (m, k, n): (usize, usize, usize),
        (lhs, lhs_storage): (&[f64], Storage),
        (rhs, rhs_storage): (&[f64], Storage),
        affine: Option<Affine<'_, f64>>,
    ) -> Vec<f64> {
        let mut out = Vec::new();
        for product in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let (alpha, beta, c) = affine.map_or((1.0, 0.0, 0.0), |affine| {
                        let [b, r, s] = affine.strides;
                        (
                            affine.alpha,
                            affine.beta,
                            affine.values[b * product + r * i + s * j],
                        )
                    });
                    let row = (0..k).map(|l| lhs[product * m * k + at(lhs_storage, m, k, i, l)]);
                    let column = (0..k).map(|l| rhs[product * k * n + at(rhs_storage, k, n, l, j)]);
                    if affine.is_none() && m * k * n > 1 << 19 {
                        let (row, column): (Vec<f64>, Vec<f64>) = (row.collect(), column.collect());
                        out.push(dot_f64(&row, &column));
                        continue;
                    }
                    let terms: Vec<[f64; 4]> = row
                        .zip(column)
                        .map(|(a, b)| [alpha, a, b, 1.0])
                        .chain([[beta, c, 1.0, 1.0]])
                        .collect();
                    out.push(exact_quads_f64(&terms) + 0.0);
                }
            }
        }
        out
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn every_entry_is_its_exact_dot_in_every_layout() {
        for (batch, m, k, n) in [
            (1, 40, 300, 37),
            (3, 16, 33, 90),
            (1, 20, 700, 24),
            (1, 132, 70, 260),
            (1, 2, 5000, 4),
            (2, 1, 40, 6),
            (2, 50, 17, 1),
            (1, 3, 4, 5),
            (1, 200, 1, 190),
        ] {
            for spread in [false, true] {
                let lhs = values(11 + m as u64, batch * m * k, spread);
                let rhs = values(29 + n as u64, batch * k * n, spread);
                for lhs_storage in [Storage::RowMajor, Storage::Transposed] {
                    for rhs_storage in [Storage::RowMajor, Storage::Transposed] {
                        let mut out = vec![f64::NAN; batch * m * n];
                        rounded_products_f64(
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
                        let want = reference(
                            batch,
                            (m, k, n),
                            (&lhs, lhs_storage),
                            (&rhs, rhs_storage),
                            None,
                        );
                        assert_eq!(
                            bits(&out),
                            bits(&want),
                            "{batch} x ({m}, {k}) @ ({k}, {n}), {lhs_storage:?} by \
                             {rhs_storage:?}, spread {spread}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_scaled_product_with_an_addend_rounds_once() {
        for (batch, m, k, n) in [(1, 30, 200, 40), (2, 1, 30, 5), (2, 4, 3, 1), (1, 5, 0, 3)] {
            let lhs = values(41 + k as u64, batch * m * k, true);
            let rhs = values(43 + n as u64, batch * k * n, false);
            let bias = values(47, n, true);
            let full = values(53, batch * m * n, false);
            for (alpha, beta) in [
                (1.0, 1.0),
                (-0.7, 2.5),
                (3.0, 0.0),
                (0.0, 1.0),
                (1e-300, 1e-310),
            ] {
                for (addend, strides) in [
                    (&bias[..], [0, 0, 1]),
                    (&full[..], [m * n, n, 1]),
                    (&full[..1], [0, 0, 0]),
                ] {
                    let affine = Affine {
                        alpha,
                        beta,
                        values: addend,
                        strides,
                    };
                    let mut out = vec![f64::NAN; batch * m * n];
                    rounded_affine_products_f64(
                        batch,
                        m,
                        k,
                        n,
                        (&lhs, Storage::RowMajor),
                        (&rhs, Storage::Transposed),
                        Some(affine),
                        &mut out,
                    )
                    .unwrap();
                    let want = reference(
                        batch,
                        (m, k, n),
                        (&lhs, Storage::RowMajor),
                        (&rhs, Storage::Transposed),
                        Some(affine),
                    );
                    assert_eq!(
                        bits(&out),
                        bits(&want),
                        "{batch} x ({m}, {k}, {n}), alpha {alpha}, beta {beta}, strides \
                         {strides:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn whole_numbers_and_zeros_are_exact_as_they_stand() {
        // An identity with a negative zero, whole numbers with a row of
        // negative zeros, and a product that cancels to zero everywhere:
        // every entry the pieces give is exact, a zero a positive one.
        let n = 48;
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        identity[3] = -0.0;
        let mut whole: Vec<f64> = (0..n * n)
            .map(|i| ((i * 7919) % 23) as f64 - 11.0)
            .collect();
        whole[..n].fill(-0.0);
        let mut out = vec![f64::NAN; n * n];
        rounded_products_f64(
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
        let want: Vec<f64> = whole.iter().map(|&v| v + 0.0).collect();
        assert_eq!(bits(&out), bits(&want));
        let cancelling: Vec<f64> = (0..n * n)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        let ones = vec![1.0; n * n];
        rounded_products_f64(
            1,
            n,
            n,
            n,
            &cancelling,
            Storage::RowMajor,
            &ones,
            Storage::RowMajor,
            &mut out,
        )
        .unwrap();
        assert_eq!(bits(&out), vec![0; n * n]);
    }

    #[test]
    fn entries_whose_terms_are_all_zeros() {
        // The product of two lower triangles: above the diagonal every term
        // of an entry is a zero, and the entry a positive one. A column of
        // the right operand then holds an infinity where the left operand's
        // rows hold zeros, and those entries are NaN, as their dots are.
        let n = 72;
        let mut lower = values(7, n * n, false);
        for i in 0..n {
            for j in i + 1..n {
                lower[i * n + j] = 0.0;
            }
        }
        let mut right = lower.clone();
        let mut out = vec![f64::NAN; n * n];
        for with_infinity in [false, true] {
            if with_infinity {
                right[(n - 1) * n + 3] = f64::INFINITY;
            }
            rounded_products_f64(
                1,
                n,
                n,
                n,
                &lower,
                Storage::RowMajor,
                &right,
                Storage::RowMajor,
                &mut out,
            )
            .unwrap();
            for i in 0..n {
                for j in 0..n {
                    let column: Vec<f64> = (0..n).map(|l| right[l * n + j]).collect();
                    let want = dot_f64(&lower[i * n..(i + 1) * n], &column);
                    let got = out[i * n + j];
                    assert!(
                        got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(),
                        "({i}, {j}): {got} where the dot is {want}"
                    );
                    if j > i && !(with_infinity && j == 3) {
                        assert_eq!(got.to_bits(), 0, "({i}, {j})");
                    }
                }
            }
        }
        assert!(out[3].is_nan() && out[n + 3].is_nan());
        assert!(no_products(&[0.0, 2.0, -0.0], &[5.0, 0.0, f64::MAX]));
        assert!(!no_products(&[0.0, 2.0], &[f64::INFINITY, 0.0]));
        assert!(!no_products(&[1e-200, 2.0], &[1e-200, 0.0]));
        assert!(!no_products(&[f64::NAN], &[0.0]));
    }

    #[test]
    fn cancellation_far_ranges_and_special_values() {
        // Rows whose dots cancel far below their terms, rows and columns past
        // the cut's range, a zero row, and rows with an infinity and a NaN,
        // in a product large enough to be cut.
        let (m, k, n) = (8, 64, 70);
        let mut lhs = values(3, m * k, false);
        let rhs = values(5, k * n, true);
        lhs[..k].fill(0.0);
        lhs[k] = 1e300;
        lhs[2 * k] = 1e-300;
        lhs[3 * k + 5] = f64::INFINITY;
        lhs[4 * k + 7] = f64::NAN;
        // Row five's last entry cancels its dot with the first column to
        // within the roundings of the rest.
        let column: Vec<f64> = (0..k).map(|l| rhs[l * n]).collect();
        let partial = dot_f64(&lhs[5 * k..6 * k - 1], &column[..k - 1]);
        lhs[6 * k - 1] = -partial / column[k - 1];
        let mut out = vec![0.0; m * n];
        rounded_products_f64(
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
        for i in 0..m {
            let row: Vec<f64> = lhs[i * k..(i + 1) * k].to_vec();
            for j in 0..n {
                let column: Vec<f64> = (0..k).map(|l| rhs[l * n + j]).collect();
                let want = dot_f64(&row, &column);
                let got = out[i * n + j];
                assert!(
                    got.to_bits() == want.to_bits() || got.is_nan() && want.is_nan(),
                    "({i}, {j}): {got} where the dot is {want}"
                );
            }
        }
        assert!(
            out[3 * n..4 * n]
                .iter()
                .all(|v| v.is_infinite() || v.is_nan())
        );
        assert_eq!(bits(&out[..n]), vec![0; n]);
    }
}
