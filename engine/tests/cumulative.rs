// Copyright (c) 2025 Soumyadip Sarkar.
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
