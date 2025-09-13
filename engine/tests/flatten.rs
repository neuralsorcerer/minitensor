// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::device::Device;
use engine::tensor::{DataType, Shape, Tensor};
use std::sync::Arc;

#[test]
fn test_flatten_basic() {
    let t = Tensor::ones(
        Shape::new(vec![2, 3, 4]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let flat = t.flatten_all().unwrap();
    assert_eq!(flat.shape().dims(), &[24]);
    assert!(Arc::ptr_eq(t.data(), flat.data()));
}

#[test]
fn test_flatten_scalar() {
    let t = Tensor::ones(Shape::scalar(), DataType::Float32, Device::cpu(), false);
    let flat = t.flatten_all().unwrap();
    assert_eq!(flat.shape().dims(), &[1]);
    assert!(Arc::ptr_eq(t.data(), flat.data()));
}

#[test]
fn test_ravel_alias() {
    let t = Tensor::zeros(
        Shape::new(vec![4, 1]),
        DataType::Int32,
        Device::cpu(),
        false,
    );
    let r = t.ravel().unwrap();
    assert_eq!(r.shape().dims(), &[4]);
    assert!(Arc::ptr_eq(t.data(), r.data()));
}

#[test]
fn test_flatten_range() {
    let t = Tensor::ones(
        Shape::new(vec![2, 3, 4]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let f = t.flatten_range(1, 2).unwrap();
    assert_eq!(f.shape().dims(), &[2, 12]);
    assert!(Arc::ptr_eq(t.data(), f.data()));
}

#[test]
fn test_flatten_range_invalid() {
    let t = Tensor::ones(
        Shape::new(vec![2, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    assert!(t.flatten_range(1, 0).is_err());
    assert!(t.flatten_range(2, 2).is_err());
}
