// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::device::Device;
use engine::ops::{arithmetic, reduction};
use engine::tensor::{DataType, Shape, Tensor, TensorData};
use std::sync::Arc;

fn tensor_f32(data: Vec<f32>, dims: Vec<usize>) -> Tensor {
    Tensor::new(
        Arc::new(TensorData::from_vec::<f32>(
            data,
            DataType::Float32,
            Device::cpu(),
        )),
        Shape::new(dims),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

#[test]
fn test_expand_basic() {
    let t = tensor_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let e = t.expand(vec![4, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[4, 3]);
    assert_eq!(
        e.data().as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_expand_neg_one() {
    let t = tensor_f32(vec![1.0, 2.0], vec![2, 1]);
    let e = t.expand(vec![-1, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[2, 3]);
    assert_eq!(
        e.data().as_f32_slice().unwrap(),
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn test_expand_adds_leading_dimensions() {
    let t = tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let e = t.expand(vec![2, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[2, 3]);
    assert_eq!(
        e.data().as_f32_slice().unwrap(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_expand_without_broadcast_shares_storage() {
    // Nothing broadcasts, so the existing buffer is already the result and
    // must not be copied.
    let t = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let e = t.expand(vec![2, 3]).unwrap();
    assert_eq!(e.shape().dims(), &[2, 3]);
    assert!(Arc::ptr_eq(t.data(), e.data()));
}

#[test]
fn test_expand_invalid() {
    let t = tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(t.expand(vec![3, 3]).is_err());
    // Fewer dimensions than the source is rejected.
    assert!(t.expand(vec![3]).is_err());
    // `-1` has no source dimension to copy for a brand-new leading axis.
    assert!(t.expand(vec![-1, 2, 3]).is_err());
}

#[test]
fn test_expand_result_feeds_elementwise_ops_correctly() {
    // Regression: `expand` used to return a stride-0 view over the original
    // (smaller) buffer, and every kernel reads storage in contiguous logical
    // order — so element-wise ops silently produced a truncated, wrong result.
    let row = tensor_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let expanded = row.expand(vec![4, 3]).unwrap();
    let other = tensor_f32(vec![10.0; 12], vec![4, 3]);

    let sum = arithmetic::add(&expanded, &other).unwrap();
    assert_eq!(sum.shape().dims(), &[4, 3]);
    assert_eq!(
        sum.data().as_f32_slice().unwrap(),
        &[
            11.0, 12.0, 13.0, 11.0, 12.0, 13.0, 11.0, 12.0, 13.0, 11.0, 12.0, 13.0
        ]
    );

    let negated = arithmetic::neg(&expanded).unwrap();
    assert_eq!(negated.data().as_f32_slice().unwrap().len(), 12);

    let total = reduction::sum(&expanded, None, false).unwrap();
    assert_eq!(total.data().as_f32_slice().unwrap()[0], 24.0);
}

#[test]
fn test_expand_zero_sized_target() {
    let t = tensor_f32(Vec::new(), vec![0]);
    let e = t.expand(vec![1, 0]).unwrap();
    assert_eq!(e.shape().dims(), &[1, 0]);
    assert!(e.data().as_f32_slice().unwrap().is_empty());
}
