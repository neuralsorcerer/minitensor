// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Reductions must return bitwise identical results regardless of how many
//! threads rayon happens to run with.
//!
//! Floating-point addition is not associative, so a reduction that lets the
//! runtime decide how partial results are grouped produces different answers on
//! machines with different core counts. `sum(dim=0)` did exactly that: it folded
//! a per-worker accumulator over rows, so the grouping followed rayon's
//! work-stealing.

use std::sync::Arc;

use engine::ops::reduction;
use engine::tensor::Shape;
use engine::{DataType, Device, Tensor, TensorData};

/// Values spanning many magnitudes, so that regrouping the additions changes
/// the rounding and a non-deterministic reduction is actually caught.
fn wide_magnitude_tensor(rows: usize, cols: usize) -> Tensor {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let mantissa = (next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exponent = ((next() >> 58) as i32) - 16;
            mantissa * 2f32.powi(exponent)
        })
        .collect();
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![rows, cols]),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

/// Well-scaled values in roughly `[-1, 1]`, for the cases that have to stay in a
/// range where arithmetic behaves — a training loop fed
/// [`wide_magnitude_tensor`] overflows to infinity within a few steps.
fn moderate_tensor(rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        .collect();
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![rows, cols]),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

/// A stack of symmetric positive-definite matrices, built as `R R^T + n I` so
/// the diagonal dominates and the factorisation is nowhere near failing.
///
/// `cholesky` splits two ways at once -- across the batch, and inside the GEMM
/// that folds each panel into the next -- so it is worth its own fixture rather
/// than reusing a rectangular one.
fn spd_batch(batch: usize, n: usize, seed: u64) -> Tensor {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut data = vec![0f32; batch * n * n];
    let mut root = vec![0f32; n * n];
    for matrix in data.chunks_mut(n * n) {
        for slot in root.iter_mut() {
            *slot = (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0f32;
                for k in 0..n {
                    acc += root[i * n + k] * root[j * n + k];
                }
                matrix[i * n + j] = acc + if i == j { n as f32 } else { 0.0 };
            }
        }
    }
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![batch, n, n]),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

fn sum_dim0_bits(tensor: &Tensor, threads: usize) -> Vec<u32> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let out = reduction::sum(tensor, Some(vec![0]), false).unwrap();
        out.data()
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    })
}

#[test]
fn test_sum_dim0_is_bitwise_stable_across_thread_counts() {
    // Both strategies are covered: enough rows to split into bands, and too few
    // rows for that, where the work is split across output columns instead.
    for (rows, cols) in [(2000usize, 300usize), (6000, 17), (3, 4096), (8, 1000)] {
        let tensor = wide_magnitude_tensor(rows, cols);
        let reference = sum_dim0_bits(&tensor, 1);
        for threads in [2usize, 3, 5, 8] {
            assert_eq!(
                sum_dim0_bits(&tensor, threads),
                reference,
                "sum(dim=0) on {rows}x{cols} changed with {threads} threads"
            );
        }
    }
}

#[test]
fn test_nansum_dim0_is_bitwise_stable_across_thread_counts() {
    let base = wide_magnitude_tensor(2000, 300);
    let mut values = base.data().as_f32_slice().unwrap().to_vec();
    for (i, value) in values.iter_mut().enumerate() {
        if i % 7 == 0 {
            *value = f32::NAN;
        }
    }
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(values, Device::cpu())),
        Shape::new(vec![2000, 300]),
        DataType::Float32,
        Device::cpu(),
        false,
    );

    let run = |threads: usize| -> Vec<u32> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            let out = reduction::nansum(&tensor, Some(vec![0]), false).unwrap();
            out.data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
    };

    let reference = run(1);
    for threads in [2usize, 4, 8] {
        assert_eq!(
            run(threads),
            reference,
            "nansum(dim=0) changed with {threads}"
        );
    }
}

/// Train a small regression for a fixed number of steps and return the bits of
/// the resulting parameters.
///
/// The bias broadcasts over the batch, so its gradient is a reduction along
/// dimension 0 -- the path that used to vary with the thread count. Running the
/// whole loop inside one `install` keeps it on a single worker, which matters
/// because the autograd graph is thread-local.
fn train_steps_bits(threads: usize) -> (Vec<u32>, Vec<u32>) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let (batch, features, outputs) = (2000usize, 6usize, 4usize);
        let x = moderate_tensor(batch, features, 0x9E37_79B9);
        let y = moderate_tensor(batch, outputs, 0x1234_5678);

        let mut w = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..features * outputs)
                    .map(|i| (i % 5) as f32 * 0.1 - 0.2)
                    .collect(),
                Device::cpu(),
            )),
            Shape::new(vec![features, outputs]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let mut b = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![0.0; outputs], Device::cpu())),
            Shape::new(vec![outputs]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let lr = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![5e-2], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        for _ in 0..12 {
            let pred =
                engine::ops::arithmetic::add(&engine::ops::linalg::matmul(&x, &w).unwrap(), &b)
                    .unwrap();
            let diff = engine::ops::arithmetic::sub(&pred, &y).unwrap();
            let sq = engine::ops::arithmetic::mul(&diff, &diff).unwrap();
            let loss = reduction::mean(&sq, None, false).unwrap();
            engine::autograd::backward(&loss, None).unwrap();

            let gw = engine::autograd::get_gradient(&w).unwrap().detach();
            let gb = engine::autograd::get_gradient(&b).unwrap().detach();
            w = engine::ops::arithmetic::sub(&w, &engine::ops::arithmetic::mul(&gw, &lr).unwrap())
                .unwrap()
                .detach()
                .requires_grad_(true);
            b = engine::ops::arithmetic::sub(&b, &engine::ops::arithmetic::mul(&gb, &lr).unwrap())
                .unwrap()
                .detach()
                .requires_grad_(true);
            engine::autograd::clear_graph().unwrap();
        }

        // Two identically-diverged runs would compare equal and prove nothing.
        // An earlier version of this test fed the loop wide-magnitude inputs
        // and every weight reached infinity within a few steps, so it passed
        // against a knowingly non-deterministic kernel. Check that the run
        // stayed finite and that the weights actually moved.
        let final_w = w.data().as_f32_slice().unwrap().to_vec();
        let final_b = b.data().as_f32_slice().unwrap().to_vec();
        assert!(
            final_w.iter().chain(final_b.iter()).all(|v| v.is_finite()),
            "training diverged; a comparison of two diverged runs is vacuous"
        );
        assert!(
            final_w
                .iter()
                .enumerate()
                .any(|(i, v)| (v - ((i % 5) as f32 * 0.1 - 0.2)).abs() > 1e-6),
            "weights never moved; the comparison would not exercise training"
        );

        let bits = |t: &Tensor| -> Vec<u32> {
            t.data()
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        (bits(&w), bits(&b))
    })
}

#[test]
fn test_training_is_bitwise_reproducible_across_thread_counts() {
    // The end-to-end property that matters: the same seed and the same steps
    // must land on the same weights whatever the machine's core count.
    let reference = train_steps_bits(1);
    for threads in [2usize, 4, 8] {
        assert_eq!(
            train_steps_bits(threads),
            reference,
            "training diverged with {threads} threads"
        );
    }
}

#[test]
fn test_sum_dim0_matches_a_sequential_reference() {
    // Determinism is only worth having if the value is also right.
    let (rows, cols) = (1500usize, 40usize);
    let tensor = wide_magnitude_tensor(rows, cols);
    let input = tensor.data().as_f32_slice().unwrap().to_vec();

    let got = reduction::sum(&tensor, Some(vec![0]), false).unwrap();
    let got = got.data().as_f32_slice().unwrap();

    for (col, &value) in got.iter().enumerate() {
        let expected: f64 = (0..rows).map(|r| input[r * cols + col] as f64).sum();
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (value as f64 - expected).abs() <= tolerance,
            "column {col}: {value} vs {expected}"
        );
    }
}

/// Run `produce` inside a pool of exactly `threads` workers and return the raw
/// bits of the result, so two runs can be compared without float equality
/// having an opinion about NaN.
fn bits_with_threads<F>(threads: usize, produce: F) -> Vec<u32>
where
    F: Fn() -> Tensor + Send + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let out = produce();
        let data = out.data();
        // Every operation below produces one of these three dtypes; the bits
        // are what matter, not the interpretation.
        if let Some(values) = data.as_f32_slice() {
            values.iter().map(|v| v.to_bits()).collect()
        } else if let Some(values) = data.as_i64_slice() {
            values.iter().map(|v| *v as u32).collect()
        } else {
            data.as_bool_slice()
                .expect("f32, i64 or bool result")
                .iter()
                .map(|v| *v as u32)
                .collect()
        }
    })
}

/// Assert that `produce` gives the same bits at every pool size.
fn assert_thread_invariant<F>(name: &str, produce: F)
where
    F: Fn() -> Tensor + Send + Sync,
{
    let reference = bits_with_threads(1, &produce);
    for threads in [2usize, 3, 5, 8] {
        assert_eq!(
            bits_with_threads(threads, &produce),
            reference,
            "{name} changed with {threads} threads"
        );
    }
}

/// Everything that splits work across the pool has to give the same answer
/// whatever the pool size, not only the sum this file was written for.
///
/// The surface here is the one whose parallel structure is load-bearing: the
/// element-wise arithmetic (split into blocks above a threshold), the
/// reductions that band their output (`any`/`all`, the fused variance), and the
/// movement kernels (which write into raw capacity, so a mis-tiled split would
/// leave uninitialised bytes rather than merely regrouping additions).
///
/// Only `sum` had a test before, because `sum` is where non-associativity bites
/// hardest — but a split that drops or double-counts an element is a bug in any
/// of them, and shows up here as differing bits rather than as a wrong total
/// nobody has a reference for.
///
/// What this can and cannot see is worth being precise about, because it was
/// checked rather than assumed. Making the dim-0 row banding follow the thread
/// count — the defect this file was originally written for — fails the
/// `mean(dim=0)` case here, which nothing covered before. Making the *fused
/// variance* band by thread count does not fail anything, and correctly so:
/// each row's variance is computed entirely within one task, so where the row
/// boundaries fall cannot regroup an addition. A kernel whose splits are
/// genuinely independent has nothing for this test to catch, and that is the
/// answer rather than a gap in it.
#[test]
fn the_parallel_kernels_are_bitwise_stable_across_thread_counts() {
    let a = wide_magnitude_tensor(700, 512);
    let b = moderate_tensor(700, 512, 0x9E37);
    let column = moderate_tensor(700, 1, 0xBEEF);

    // Element-wise: equal-shape (blocked fast path) and broadcasting.
    assert_thread_invariant("add", || engine::ops::arithmetic::add(&a, &b).unwrap());
    assert_thread_invariant("mul", || engine::ops::arithmetic::mul(&a, &b).unwrap());
    assert_thread_invariant("div", || engine::ops::arithmetic::div(&a, &b).unwrap());
    assert_thread_invariant("add broadcast", || {
        engine::ops::arithmetic::add(&a, &column).unwrap()
    });

    // Cheap unary, including the multiversioned rounding family.
    let positive = engine::ops::activation::abs(&a).unwrap();
    assert_thread_invariant("abs", || engine::ops::activation::abs(&a).unwrap());
    assert_thread_invariant("floor", || engine::ops::activation::floor(&a).unwrap());
    assert_thread_invariant("round", || engine::ops::activation::round(&a, 0).unwrap());
    assert_thread_invariant("reciprocal", || {
        engine::ops::activation::reciprocal(&positive).unwrap()
    });
    assert_thread_invariant("clip", || {
        engine::ops::activation::clip(&a, Some(-0.5), Some(0.5)).unwrap()
    });
    assert_thread_invariant("isnan", || a.isnan().unwrap());

    // Reductions on both axes: the two layouts are different code.
    for dim in [0isize, 1] {
        assert_thread_invariant(&format!("mean(dim={dim})"), || {
            reduction::mean(&a, Some(vec![dim]), false).unwrap()
        });
        assert_thread_invariant(&format!("prod(dim={dim})"), || {
            reduction::prod(&b, Some(vec![dim]), false).unwrap()
        });
        assert_thread_invariant(&format!("var(dim={dim})"), || {
            reduction::var(&a, Some(vec![dim]), false, false).unwrap()
        });
        assert_thread_invariant(&format!("all(dim={dim})"), || {
            reduction::all(&a, Some(dim), false).unwrap()
        });
        assert_thread_invariant(&format!("any(dim={dim})"), || {
            reduction::any(&a, Some(dim), false).unwrap()
        });
        assert_thread_invariant(&format!("cumsum(dim={dim})"), || {
            reduction::cumsum(&a, dim).unwrap()
        });
    }

    // Movement: these write into uninitialised capacity, so a split that fails
    // to tile the output shows up as bits that differ run to run.
    assert_thread_invariant("flip", || engine::ops::shape_ops::flip(&a, &[0]).unwrap());
    assert_thread_invariant("roll", || {
        engine::ops::shape_ops::roll(&a, &[7], Some(&[1])).unwrap()
    });
    assert_thread_invariant("concatenate", || {
        engine::ops::shape_ops::concatenate(&[&a, &b], 0).unwrap()
    });
    assert_thread_invariant("repeat", || {
        engine::ops::shape_ops::repeat(&a, &[2, 1]).unwrap()
    });
    assert_thread_invariant("slice", || {
        engine::ops::shape_ops::slice(&a, 0, 100, 600, 3).unwrap()
    });

    // The factorisation splits across the batch *and* inside the GEMM that
    // folds each finished panel into the next, so both splits are on trial
    // here. The wide matrix is deliberately past one panel: a single-panel
    // matrix never reaches the GEMM at all and would test only the batching.
    let narrow = spd_batch(24, 12, 0x51ED);
    let wide = spd_batch(3, 150, 0xC0FF);
    assert_thread_invariant("cholesky (batched, one panel)", || {
        engine::ops::linalg::cholesky(&narrow, false).unwrap()
    });
    assert_thread_invariant("cholesky (panelled)", || {
        engine::ops::linalg::cholesky(&wide, false).unwrap()
    });

    // `qr` splits across the batch and, above a size threshold, folds each
    // panel into the trailing block with GEMMs instead of one reflector at a
    // time. Both shapes below are here so both paths are on trial: the small
    // one never reaches the blocked update, the tall one does.
    let short = moderate_tensor(40, 9, 0x1234);
    let tall = moderate_tensor(300, 120, 0xFEED);
    for (name, matrix) in [("qr (direct)", &short), ("qr (blocked)", &tall)] {
        assert_thread_invariant(&format!("{name} Q"), || {
            engine::ops::linalg::qr(matrix, engine::ops::linalg::QrMode::Reduced)
                .unwrap()
                .0
        });
        assert_thread_invariant(&format!("{name} R"), || {
            engine::ops::linalg::qr(matrix, engine::ops::linalg::QrMode::Reduced)
                .unwrap()
                .1
        });
    }
}
