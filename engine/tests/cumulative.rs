// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use std::sync::Arc;

use engine::tensor::Shape;
use engine::{DataType, Device, Tensor, TensorData};

#[test]
fn test_cumsum_and_cumprod_forward() {
    let data = Arc::new(TensorData::from_vec_f32(
        (1..=6).map(|v| v as f32).collect(),
        Device::cpu(),
    ));
    let t = Tensor::new(
        data,
        Shape::new(vec![2, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );

    let c0 = t.cumsum(0).unwrap();
    assert_eq!(
        c0.data().as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    );

    let c1 = t.cumsum(1).unwrap();
    assert_eq!(
        c1.data().as_f32_slice().unwrap(),
        &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );

    let p0 = t.cumprod(0).unwrap();
    assert_eq!(
        p0.data().as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 18.0]
    );

    let p1 = t.cumprod(1).unwrap();
    assert_eq!(
        p1.data().as_f32_slice().unwrap(),
        &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0]
    );
}

/// Naive reference scan over a row-major buffer: for every position, walk back
/// along `dim` accumulating from the start of that line.
fn reference_scan(data: &[f32], dims: &[usize], dim: usize, product: bool) -> Vec<f32> {
    let dim_size = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();
    let mut out = vec![0.0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = if product { 1.0 } else { 0.0 };
            for d in 0..dim_size {
                let idx = (o * dim_size + d) * inner + i;
                acc = if product {
                    acc * data[idx]
                } else {
                    acc + data[idx]
                };
                out[idx] = acc;
            }
        }
    }
    out
}

fn tensor_f32(data: Vec<f32>, dims: Vec<usize>) -> Tensor {
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(dims),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

#[test]
fn test_cumulative_matches_reference_across_ranks_and_dims() {
    // The scan kernel is shared by every rank and dimension, including the
    // sizes that cross into the parallel per-slab path, so check them all
    // against an independent implementation.
    let shapes: &[Vec<usize>] = &[
        vec![7],
        vec![1],
        vec![3, 4],
        vec![4, 3],
        vec![2, 3, 4],
        vec![5, 1, 6],
        vec![2, 3, 4, 5],
        vec![64, 128], // > PAR_THRESHOLD, exercises the slab-parallel path
        vec![8, 16, 64],
    ];
    for shape in shapes {
        let numel: usize = shape.iter().product();
        // Values near 1 so cumprod stays in a sane range for the long axes.
        let data: Vec<f32> = (0..numel).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
        let t = tensor_f32(data.clone(), shape.clone());
        for dim in 0..shape.len() {
            let got_sum = t.cumsum(dim as isize).unwrap();
            let want_sum = reference_scan(&data, shape, dim, false);
            assert_eq!(
                got_sum.data().as_f32_slice().unwrap(),
                want_sum.as_slice(),
                "cumsum shape={shape:?} dim={dim}"
            );

            let got_prod = t.cumprod(dim as isize).unwrap();
            let want_prod = reference_scan(&data, shape, dim, true);
            let got_prod = got_prod.data().as_f32_slice().unwrap();
            for (i, (&g, &w)) in got_prod.iter().zip(want_prod.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "cumprod shape={shape:?} dim={dim} index {i}: {g} vs {w}"
                );
            }
        }
    }
}

#[test]
fn test_cumulative_handles_zero_sized_tensors() {
    let t = tensor_f32(Vec::new(), vec![0, 3]);
    assert!(
        t.cumsum(0)
            .unwrap()
            .data()
            .as_f32_slice()
            .unwrap()
            .is_empty()
    );
    assert!(
        t.cumsum(1)
            .unwrap()
            .data()
            .as_f32_slice()
            .unwrap()
            .is_empty()
    );
    assert!(
        t.cumprod(0)
            .unwrap()
            .data()
            .as_f32_slice()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn test_cumsum_dim_out_of_bounds() {
    let data = Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu()));
    let t = Tensor::new(
        data,
        Shape::new(vec![3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    assert!(t.cumsum(1).is_err());
    assert!(t.cumprod(1).is_err());
}
