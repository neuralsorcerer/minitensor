// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The gradients a product or a quotient hands an operand it broadcast,
//! each rounded once.
//!
//! An operand read at several positions gathers, at each of its elements, the
//! gradient of every position it was read at: for a factor, the dot of the
//! incoming gradient with the other factor over those positions; for a
//! divisor, which is the one value at all of them, minus that dot with the
//! numerator over its square. Each was its terms rounded and then added.
//! Here each is the dot carried -- over the divisor's square, an exact pair
//! -- and rounded once, in whole numbers where the bound does not settle it.

use super::{Carried, Element, Output, Total, ratio_settled, two_sum};
use std::mem::MaybeUninit;

/// For each of `groups` runs of `run` positions, `a` and `b` laid out a run
/// at a time, `dot` of the run's values: zero for a run of none.
pub(crate) fn group_dots<T: Element + Output>(
    (a, b): (&[T], &[T]),
    (run, groups): (usize, usize),
    dot: fn(&[T], &[T]) -> T,
) -> Vec<T> {
    grouped(groups, run, |k| {
        let span = k * run..(k + 1) * run;
        dot(&a[span.clone()], &b[span])
    })
}

/// For each of the runs of `run` positions, `g` and `x` laid out a run at a
/// time, the gradient of `x / y` in its divisor: `-(Σ g x) / y[k]^2` over
/// the run, the value of `T` nearest it. Where a value is not finite or the
/// divisor is zero, it is the formula's. A float64 run's products' errors
/// are fused multiply-adds where the CPU has them, the same to the bit.
pub(crate) fn divisor_slopes<T: Element + Output>(
    (g, x): (&[T], &[T]),
    y: &[T],
    run: usize,
) -> Vec<T> {
    let fma = !cfg!(target_arch = "x86_64") || super::fused();
    let chunk = (8192 / run.max(1)).max(1);
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, chunk, &|start, part| {
            let span = start * run..(start + part.len()) * run;
            let operands = (&g[span.clone()], &x[span], &y[start..start + part.len()]);
            if fma && !T::PLAIN {
                // SAFETY: where `fma` the CPU has avx2 and fma, or the target a
                // fused multiply-add of its own.
                #[allow(unused_unsafe)]
                return unsafe { ratios_fma(operands, run, part) };
            }
            ratios::<T, false>(operands, run, part);
        });
    };
    // SAFETY: the chunks cover the output, and a run writes each element.
    unsafe { crate::ops::map::build_vec(y.len(), fill) }
}

/// [`ratios`] compiled for fused multiply-adds.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
fn ratios_fma<T: Element + Output>(
    operands: (&[T], &[T], &[T]),
    run: usize,
    out: &mut [MaybeUninit<T>],
) {
    ratios::<T, true>(operands, run, out);
}

/// A part of [`divisor_slopes`]. For float32 with a term a position, first
/// the plain quotient of its two products, each exact, which rounds once:
/// a pass without branches, and then the terms it does not settle.
#[inline(always)]
fn ratios<T: Element + Output, const FMA: bool>(
    (g, x, y): (&[T], &[T], &[T]),
    run: usize,
    out: &mut [MaybeUninit<T>],
) {
    if run == 0 {
        out.iter_mut().for_each(|cell| {
            cell.write(T::narrow(0.0));
        });
        return;
    }
    // The plain quotient and whether its bound settles it.
    let plain = |a: T, b: T, c: T| {
        let value = -(a.wide() * b.wide()) / (c.wide() * c.wide());
        let (narrow, clear) = T::settle_lanes(value, 0.0, value.abs() * 2f64.powi(-52));
        let size = value.abs();
        let ordinary = (size == 0.0) | ((size >= 2f64.powi(-900)) & (size <= 2f64.powi(900)));
        (narrow, clear & ordinary)
    };
    // For float64 with fused multiply-adds, the quotient of the two exact
    // products, its exact remainder, and what is left over the divisor's
    // square, which the bound allows eight roundings for and takes four.
    let pair = |a: T, b: T, c: T| {
        let (a, b, c) = (a.wide(), b.wide(), c.wide());
        let (p, s) = (a * b, c * c);
        let (e, es) = (a.mul_add(b, -p), c.mul_add(c, -s));
        let q = p / s;
        let r = (-q).mul_add(s, p);
        let t = ((r + e) - q * es) / s;
        let (value, rest) = two_sum(-q, -t);
        let bound = ((r.abs() + e.abs() + (q * es).abs()) / s + t.abs()) * 2f64.powi(-50);
        let (narrow, clear) = T::settle_lanes(value, rest, bound);
        let in_range = |v: f64| (v.abs() >= 2f64.powi(-900)) & (v.abs() <= 2f64.powi(900));
        (narrow, clear & in_range(p) & in_range(s) & in_range(q))
    };
    if (T::PLAIN || FMA) && run == 1 {
        let pass = |a, b, c| {
            if T::PLAIN {
                plain(a, b, c)
            } else {
                pair(a, b, c)
            }
        };
        let mut rest = false;
        for (((cell, &a), &b), &c) in out.iter_mut().zip(g).zip(x).zip(y) {
            let (value, settled) = pass(a, b, c);
            cell.write(value);
            rest |= !settled;
        }
        if rest {
            for (k, cell) in out.iter_mut().enumerate() {
                if !pass(g[k], x[k], y[k]).1 {
                    cell.write(negated_ratio::<T, FMA>(
                        &g[k..k + 1],
                        &x[k..k + 1],
                        y[k].wide(),
                    ));
                }
            }
        }
        return;
    }
    for (k, cell) in out.iter_mut().enumerate() {
        let span = k * run..(k + 1) * run;
        cell.write(negated_ratio::<T, FMA>(
            &g[span.clone()],
            &x[span],
            y[k].wide(),
        ));
    }
}

/// `each(k)` for each of `groups` runs of `run` positions, side by side:
/// zero where a run has none.
fn grouped<T: Output>(groups: usize, run: usize, each: impl Fn(usize) -> T + Sync) -> Vec<T> {
    // Some 8192 positions a task.
    let chunk = (8192 / run.max(1)).max(1);
    let fill = |out: &mut [MaybeUninit<T>]| {
        crate::ops::map::par_out_chunks(out, chunk, &|start, part| {
            for (k, cell) in part.iter_mut().enumerate() {
                cell.write(if run == 0 {
                    T::narrow(0.0)
                } else {
                    each(start + k)
                });
            }
        });
    };
    // SAFETY: the chunks cover the output, and each element is written.
    unsafe { crate::ops::map::build_vec(groups, fill) }
}

/// `-(Σ g x) / y^2`: the dot carried, settled over the exact pair `y^2`, and
/// in whole numbers where the bound cannot say -- or where a product is
/// small enough that its error might have been rounded.
#[inline(always)]
fn negated_ratio<T: Element + Output, const FMA: bool>(g: &[T], x: &[T], y: f64) -> T {
    let error =
        |a: f64, b: f64, product: f64| crate::ops::simd::product_error::<FMA>(a, b, product);
    let pairs = || g.iter().zip(x).map(|(&a, &b)| (a.wide(), b.wide()));
    let (mut finite, mut tiny) = (y.is_finite() && y != 0.0, false);
    for (a, b) in pairs() {
        finite &= a.is_finite() & b.is_finite();
        tiny |= (a != 0.0) & (b != 0.0) & ((a * b).abs() < 2f64.powi(-968));
    }
    if !finite {
        let dot: f64 = pairs().map(|(a, b)| a * b).sum();
        return T::narrow(-dot / (y * y));
    }
    let square = y * y;
    let numerator = if let ([a], [b]) = (g, x) {
        let (a, b) = (a.wide(), b.wide());
        let sum = a * b;
        if sum == 0.0 {
            // Signed as the formula signs it.
            return T::narrow(-sum / square);
        }
        Carried {
            sum,
            error: error(a, b, sum),
            slack: 0.0,
        }
    } else {
        T::carried_dot(g, x)
    };
    let divisor = (square, error(y, y, square));
    match (!tiny).then(|| ratio_settled::<T, FMA>(numerator, divisor)) {
        Some(Some(value)) => T::narrow(-value.widen()),
        _ => exact_negated_ratio(g, x, y),
    }
}

/// [`negated_ratio`] in whole numbers.
#[cold]
#[inline(never)]
fn exact_negated_ratio<T: Element + Output>(g: &[T], x: &[T], y: f64) -> T {
    let terms: Vec<(f64, f64, f64)> = g
        .iter()
        .zip(x)
        .map(|(&a, &b)| (-a.wide(), b.wide(), 1.0))
        .collect();
    Total::of_terms(&terms).quotient::<T>(Total::of_terms(&[(y, y, 1.0)]))
}
