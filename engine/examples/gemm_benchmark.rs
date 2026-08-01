// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Times square `matmul` so you can decide whether `--features blas` is worth
//! it on your machine.
//!
//! ```text
//! cargo run --release --example gemm_benchmark
//! cargo run --release --features blas --example gemm_benchmark
//! ```
//!
//! The default path is `matrixmultiply`, which is pure Rust and needs no system
//! library; `blas` links whatever OpenBLAS is installed. Which one wins depends
//! on the machine and the size -- the gap is widest for large matrices, and for
//! small ones the two are close enough that the extra build dependency may not
//! be worth it. Measure before enabling.

use engine::tensor::{DataType, Shape, TensorData};
use engine::{Device, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn square(n: usize) -> Tensor {
    // Deterministic, non-trivial values: a matrix of zeros would let a
    // sufficiently clever BLAS skip work a real workload cannot.
    let data: Vec<f32> = (0..n * n).map(|i| (i % 97) as f32 * 0.01).collect();
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![n, n]),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

fn main() {
    let backend = if cfg!(feature = "blas") {
        "OpenBLAS (--features blas)"
    } else {
        "matrixmultiply (default)"
    };
    println!("GEMM backend: {backend}\n");
    println!("{:>12}  {:>10}  {:>12}", "size", "time", "throughput");

    for &n in &[128usize, 256, 512, 1024] {
        let a = square(n);
        let b = square(n);

        // One untimed pass so allocation and any first-call setup in the
        // backend are not charged to the measurement.
        let _ = a.matmul(&b).expect("matmul should succeed");

        let reps = if n <= 256 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..reps {
            let _ = a.matmul(&b).expect("matmul should succeed");
        }
        let seconds = start.elapsed().as_secs_f64() / reps as f64;

        // 2*n^3 flops for an n x n multiply: one multiply and one add per term.
        let gflops = 2.0 * (n as f64).powi(3) / seconds / 1e9;
        println!(
            "{:>6} x {:<5} {:>8.3} ms  {gflops:>7.2} GFLOP/s",
            n,
            n,
            seconds * 1e3
        );
    }
}
