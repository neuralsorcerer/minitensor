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
