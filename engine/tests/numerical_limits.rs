// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    device::Device,
    ops::{activation, arithmetic, comparison, loss, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

#[test]
fn test_max_min_with_nan_propagates() {
    let data = vec![f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY];
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![4]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let max_t = reduction::max(&tensor, None, false).unwrap();
    let min_t = reduction::min(&tensor, None, false).unwrap();
    let max_val = max_t.data().as_f32_slice().unwrap()[0];
    let min_val = min_t.data().as_f32_slice().unwrap()[0];
    assert!(max_val.is_nan());
    assert!(min_val.is_nan());
}

#[test]
fn test_max_min_inf_without_nan() {
    // Without NaN, ±inf reduce normally.
    let data = vec![1.0, f32::INFINITY, f32::NEG_INFINITY];
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(vec![3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let max_val = reduction::max(&tensor, None, false)
        .unwrap()
        .data()
        .as_f32_slice()
        .unwrap()[0];
    let min_val = reduction::min(&tensor, None, false)
        .unwrap()
        .data()
        .as_f32_slice()
        .unwrap()[0];
    assert!(max_val.is_infinite() && max_val.is_sign_positive());
    assert!(min_val.is_infinite() && min_val.is_sign_negative());
}

#[test]
fn test_division_by_zero_infinities() {
    let a = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1.0, -1.0], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let b = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![0.0, 0.0], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = arithmetic::div(&a, &b).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!(vals[0].is_infinite() && vals[0].is_sign_positive());
    assert!(vals[1].is_infinite() && vals[1].is_sign_negative());
}

#[test]
fn test_nan_addition_propagates() {
    let a = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![f32::NAN, 1.0], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let b = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1.0, 1.0], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = arithmetic::add(&a, &b).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!(vals[0].is_nan());
    assert_eq!(vals[1], 2.0);
}

#[test]
fn test_mul_overflow_results_infinity() {
    let a = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![3.4e38], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let b = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![10.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = arithmetic::mul(&a, &b).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_infinite());
}

#[test]
fn test_log_negative_returns_nan() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![-1.0, -2.0], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::log(&tensor).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!(vals[0].is_nan());
    assert!(vals[1].is_nan());
}

#[test]
fn test_exp_extreme_values() {
    let large = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1000.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let small = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![-1000.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let large_exp = activation::exp(&large).unwrap();
    let small_exp = activation::exp(&small).unwrap();
    let l = large_exp.data().as_f32_slice().unwrap()[0];
    let s = small_exp.data().as_f32_slice().unwrap()[0];
    assert!(l.is_infinite() && l.is_sign_positive());
    assert_eq!(s, 0.0);
}

#[test]
fn test_sqrt_negative_results_nan() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![-1.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::sqrt(&tensor).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_nan());
}

#[test]
fn test_inf_sub_inf_produces_nan() {
    let a = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![f32::INFINITY], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let b = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![f32::INFINITY], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = arithmetic::sub(&a, &b).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_nan());
}

#[test]
fn test_sigmoid_extreme_inputs() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, -1000.0],
            Device::cpu(),
        )),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::sigmoid(&tensor).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!(vals[1].abs() < 1e-6);
}

#[test]
fn test_sigmoid_inf_and_nan_inputs() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            Device::cpu(),
        )),
        Shape::new(vec![3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::sigmoid(&tensor).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 0.0);
    assert!(vals[2].is_nan());
}

#[test]
fn test_tanh_extreme_inputs() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, -1000.0],
            Device::cpu(),
        )),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::tanh(&tensor).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[1] + 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_extreme_range() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, -1000.0],
            Device::cpu(),
        )),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::softmax(&tensor, None).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!(vals[1] < 1e-6);
    assert!((vals[0] + vals[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_log_zero_returns_neg_infinity() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![0.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::log(&tensor).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_infinite() && val.is_sign_negative());
}

#[test]
fn test_sum_nan_propagates() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1.0, f32::NAN], Device::cpu())),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = reduction::sum(&tensor, None, false).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_nan());
}

#[test]
fn test_sum_overflow_results_infinity() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![f32::MAX, f32::MAX],
            Device::cpu(),
        )),
        Shape::new(vec![2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = reduction::sum(&tensor, None, false).unwrap();
    let val = result.data().as_f32_slice().unwrap()[0];
    assert!(val.is_infinite());
}

#[test]
fn test_nan_comparisons_false() {
    let a = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![f32::NAN], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let b = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![5.0], Device::cpu())),
        Shape::new(vec![1]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let lt_res = comparison::lt(&a, &b).unwrap();
    let eq_res = comparison::eq(&a, &a).unwrap();
    assert!(!lt_res.data().as_bool_slice().unwrap()[0]);
    assert!(!eq_res.data().as_bool_slice().unwrap()[0]);
}

#[test]
fn test_relu_negative_and_nan() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![-1.0, f32::NAN, 2.0],
            Device::cpu(),
        )),
        Shape::new(vec![3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let result = activation::relu(&tensor).unwrap();
    let vals = result.data().as_f32_slice().unwrap();
    assert_eq!(vals[0], 0.0);
    assert!(vals[1].is_nan());
    assert_eq!(vals[2], 2.0);
}

#[test]
fn test_log_softmax_extreme_range() {
    let tensor = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, -1000.0, 0.0],
            Device::cpu(),
        )),
        Shape::new(vec![3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let softmax = activation::softmax(&tensor, None).unwrap();
    let log_sm = activation::log(&softmax).unwrap();
    let vals = log_sm.data().as_f32_slice().unwrap();
    let exp_sum: f32 = vals.iter().map(|v| v.exp()).sum();
    assert!((exp_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_cross_entropy_underflowed_probability_has_finite_loss() {
    let pred = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, -1000.0],
            Device::cpu(),
        )),
        Shape::new(vec![1, 2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let target = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![0.0, 1.0], Device::cpu())),
        Shape::new(vec![1, 2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let loss = loss::cross_entropy(&pred, &target, "mean", 1).unwrap();
    let val = loss.data().as_f32_slice().unwrap()[0];
    assert!(val.is_finite());
    assert!((val - 2000.0).abs() < 1e-3, "expected 2000, got {val}");
}

#[test]
fn test_cross_entropy_ignores_non_target_negative_infinity() {
    let pred = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![0.0, f32::NEG_INFINITY],
            Device::cpu(),
        )),
        Shape::new(vec![1, 2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let correct_target = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1.0, 0.0], Device::cpu())),
        Shape::new(vec![1, 2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let impossible_target = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![0.0, 1.0], Device::cpu())),
        Shape::new(vec![1, 2]),
        DataType::Float32,
        Device::cpu(),
        false,
    );

    let correct = loss::cross_entropy(&pred, &correct_target, "mean", 1).unwrap();
    assert_eq!(correct.data().as_f32_slice().unwrap()[0], 0.0);

    let impossible = loss::cross_entropy(&pred, &impossible_target, "mean", 1).unwrap();
    let impossible_value = impossible.data().as_f32_slice().unwrap()[0];
    assert!(impossible_value.is_infinite() && impossible_value.is_sign_positive());
}

#[test]
fn test_cross_entropy_confident_correct_prediction_is_finite() {
    // Target is the argmax class (logit 1000) with a large gap: loss must be
    // ~0, not NaN. Masking every near-zero-probability class to -inf made the
    // non-target class contribute 0 * -inf = NaN before this was restricted to
    // target classes only.
    let pred = Tensor::new(
        Arc::new(TensorData::from_vec_f32(
            vec![1000.0, 0.0, -1000.0],
            Device::cpu(),
        )),
        Shape::new(vec![1, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let target = Tensor::new(
        Arc::new(TensorData::from_vec_f32(vec![1.0, 0.0, 0.0], Device::cpu())),
        Shape::new(vec![1, 3]),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let loss = loss::cross_entropy(&pred, &target, "mean", 1).unwrap();
    let val = loss.data().as_f32_slice().unwrap()[0];
    assert!(val.is_finite(), "loss must be finite, got {val}");
    assert!(val.abs() < 1e-4, "loss must be ~0, got {val}");
}
