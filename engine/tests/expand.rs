// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::device::Device;
use engine::tensor::{DataType, Shape, Tensor};
use std::sync::Arc;

#[test]
fn test_expand_basic() {
    let t = Tensor::ones(
        Shape::new(vec![1, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let e = t.expand(vec![4, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[4, 3]);
    assert!(Arc::ptr_eq(t.data(), e.data()));
}

#[test]
fn test_expand_neg_one() {
    let t = Tensor::ones(
        Shape::new(vec![2, 1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let e = t.expand(vec![-1, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[2, 3]);
    assert!(Arc::ptr_eq(t.data(), e.data()));
}

#[test]
fn test_expand_invalid() {
    let t = Tensor::ones(
        Shape::new(vec![2, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    assert!(t.expand(vec![3, 3]).is_err());
}
