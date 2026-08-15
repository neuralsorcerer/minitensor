// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Guards against an operation quietly falling off its fast path.
//!
//! Every performance defect found in this engine so far was invisible to the
//! correctness suite and to any absolute timing, because the op was *correct*
//! and the machine's speed is not fixed. Each one showed up the same way
//! instead: as a ratio against a sibling operation doing comparable work on the
//! same data. `floor` cost twenty times `abs`. `reciprocal` cost twenty-five.
//! An equal-shape `a + b` cost four times the broadcasting form of itself.
//! `all` along a dimension cost eleven times `sum` along the same one.
//!
//! A ratio is the right thing to assert because both sides scale together: a
//! slower machine, a loaded CI runner or a debug build move numerator and
//! denominator alike. What does not move is an op that has stopped vectorizing,
//! stopped parallelizing, or started allocating per row — and those are exactly
//! the regressions this is here to catch.
//!
//! The bounds are deliberately loose. Every ratio below currently sits between
//! 0.7 and 3, and the defects these replace measured 4x to 25x, so a bound of
//! 4 to 6 has room for a noisy runner while still catching anything that falls
//! off a path. A failure here is not "this got a bit slower"; it means an
//! operation is doing categorically different work than its neighbour.

use engine::device::Device;
use engine::ops::{activation, arithmetic, reduction, shape_ops};
use engine::tensor::{DataType, Shape, Tensor, TensorData};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn tensor(dims: Vec<usize>, seed: f32) -> Tensor {
    let shape = Shape::new(dims);
    let n = shape.numel();
    let data: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 + seed).collect();
    Tensor::new(
        Arc::new(TensorData::from_vec::<f32>(
            data,
            DataType::Float32,
            Device::cpu(),
        )),
        shape,
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

/// Best-of-`n` wall time, which is the statistic that survives a shared runner:
/// interference can only ever make a sample slower, so the minimum is the
/// closest thing to the work itself.
fn best<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..25 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Assert `subject` costs no more than `bound` times `sibling`.
fn ratio_within(name: &str, bound: f64, mut subject: impl FnMut(), mut sibling: impl FnMut()) {
    let s = best(&mut subject);
    let r = best(&mut sibling);
    let ratio = s / r;
    assert!(
        ratio <= bound,
        "{name}: {:.1}us against the sibling's {:.1}us is {ratio:.2}x, over the {bound}x bound. \
         An operation this much dearer than its neighbour on the same data has stopped \
         vectorizing, stopped parallelizing, or started allocating per element -- it is not \
         doing the same kind of work any more.",
        s * 1e6,
        r * 1e6
    );
}

/// Perf ratios only mean anything against optimized code: an unoptimized build
/// leaves every loop scalar, so the very distinctions being measured are gone.
fn skip_unoptimized() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipping fast-path ratios: needs an optimized build (cargo test --release)");
        return true;
    }
    false
}

/// The cheap unary ops all read one float and write one. They must cost about
/// the same; `abs` is the reference because it has nothing that can go wrong.
///
/// `floor`, `ceil` and `round` need `roundps`, which is not in the x86-64
/// baseline. Left to the baseline, LLVM cannot vectorize them at all and emits
/// a `libm` call per element, which measured 20x `abs`. `reciprocal` reached
/// `powf` through a runtime exponent and measured 25x. `clamp` opened with a
/// NaN branch that kept the loop scalar and measured 9x.
#[test]
fn the_cheap_unary_ops_cost_about_what_abs_costs() {
    if skip_unoptimized() {
        return;
    }
    let x = tensor(vec![1 << 20], 0.5);
    let reference = || {
        black_box(activation::abs(&x).unwrap());
    };
    ratio_within(
        "floor",
        4.0,
        || {
            black_box(activation::floor(&x).unwrap());
        },
        reference,
    );
    ratio_within(
        "ceil",
        4.0,
        || {
            black_box(activation::ceil(&x).unwrap());
        },
        reference,
    );
    ratio_within(
        "round",
        4.0,
        || {
            black_box(activation::round(&x, 0).unwrap());
        },
        reference,
    );
    ratio_within(
        "reciprocal",
        4.0,
        || {
            black_box(activation::reciprocal(&x).unwrap());
        },
        reference,
    );
    ratio_within(
        "clamp",
        4.0,
        || {
            black_box(activation::clip(&x, Some(0.2), Some(0.8)).unwrap());
        },
        reference,
    );
    ratio_within(
        "isnan",
        4.0,
        || {
            black_box(x.isnan().unwrap());
        },
        reference,
    );
}

/// Adding two tensors of the same shape must not cost more than adding a
/// tensor to a column that broadcasts over it. The broadcasting form does
/// strictly more per element -- it walks an index -- so the equal-shape form is
/// the one with room to be quicker, never dearer.
///
/// It was four times dearer: the equal-shape path handed the whole buffer to
/// one sequential call while the broadcasting path split across the pool, so
/// giving an operand a dimension of 1 made the operation faster.
#[test]
fn an_equal_shape_add_is_no_dearer_than_a_broadcasting_one() {
    if skip_unoptimized() {
        return;
    }
    let m = tensor(vec![1024, 1024], 0.5);
    let same = tensor(vec![1024, 1024], 1.5);
    let column = tensor(vec![1024, 1], 1.5);
    ratio_within(
        "equal-shape add against broadcast add",
        3.0,
        || {
            black_box(arithmetic::add(&m, &same).unwrap());
        },
        || {
            black_box(arithmetic::add(&m, &column).unwrap());
        },
    );
}

/// `all` and `sum` walk the same elements along the same axis; `all` writes
/// less and may stop early, so it has no business costing much more.
///
/// It cost eleven times as much along dimension 0, having scanned one output at
/// a time -- striding the input and touching a fresh cache line every step --
/// where `sum` accumulates whole slabs in memory order.
#[test]
fn a_boolean_fold_costs_about_what_a_sum_along_the_same_axis_costs() {
    if skip_unoptimized() {
        return;
    }
    let m = tensor(vec![1024, 1024], 0.5);
    for dim in [0usize, 1] {
        ratio_within(
            &format!("all(dim={dim}) against sum(dim={dim})"),
            6.0,
            || {
                black_box(reduction::all(&m, Some(dim as isize), false).unwrap());
            },
            || {
                black_box(reduction::sum(&m, Some(vec![dim as isize]), false).unwrap());
            },
        );
    }
}

/// `flip` and `roll` relocate exactly the same bytes as each other, one row at
/// a time. Neither computes anything, so they must cost the same; a gap means
/// one of them grew a pass over its output that the other did not.
///
/// Both grew one: they allocated a zeroed buffer and then overwrote every byte
/// of it, which was most of a memcpy-bound operation.
#[test]
fn the_two_row_relocations_cost_the_same() {
    if skip_unoptimized() {
        return;
    }
    let m = tensor(vec![1024, 1024], 0.5);
    ratio_within(
        "flip against roll",
        3.0,
        || {
            black_box(shape_ops::flip(&m, &[0]).unwrap());
        },
        || {
            black_box(shape_ops::roll(&m, &[7], Some(&[1])).unwrap());
        },
    );
}

/// Variance reduces the same elements a mean does, in two passes rather than
/// one, so a handful of times a mean is expected and an order of magnitude is
/// not. Reducing the *last* axis is the case that went wrong: its outputs are
/// one element apart, and chunking the output by that stride handed rayon one
/// task per row, each allocating a one-element scratch buffer on the heap.
#[test]
fn variance_stays_within_a_few_means() {
    if skip_unoptimized() {
        return;
    }
    let m = tensor(vec![1024, 1024], 0.5);
    for dim in [0usize, 1] {
        ratio_within(
            &format!("var(dim={dim}) against mean(dim={dim})"),
            14.0,
            || {
                black_box(reduction::var(&m, Some(vec![dim as isize]), false, false).unwrap());
            },
            || {
                black_box(reduction::mean(&m, Some(vec![dim as isize]), false).unwrap());
            },
        );
    }
}
