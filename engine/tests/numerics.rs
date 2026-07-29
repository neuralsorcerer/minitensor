// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Value semantics that have to agree with NumPy and PyTorch, in places where a
//! plausible implementation quietly disagrees.

use std::sync::Arc;

use engine::ops::activation::round;
use engine::tensor::Shape;
use engine::{DataType, Device, Tensor, TensorData};

fn tensor_f32(values: Vec<f32>) -> Tensor {
    let len = values.len();
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(values, Device::cpu())),
        Shape::new(vec![len]),
        DataType::Float32,
        Device::cpu(),
        false,
    )
}

fn tensor_f64(values: Vec<f64>) -> Tensor {
    let len = values.len();
    Tensor::new(
        Arc::new(TensorData::from_vec_f64(values, Device::cpu())),
        Shape::new(vec![len]),
        DataType::Float64,
        Device::cpu(),
        false,
    )
}

#[test]
fn test_round_breaks_ties_to_even() {
    // NumPy, PyTorch and Python's built-in `round` all send halves to the even
    // neighbour. Rust's `f32::round` sends them away from zero, so calling it
    // directly disagreed at every exact .5: round(0.5) gave 1 rather than 0,
    // round(2.5) gave 3 rather than 2.
    let input = vec![-3.5f32, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    let expected = [-4.0f32, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0];

    let got = round(&tensor_f32(input.clone()), 0).unwrap();
    assert_eq!(got.data().as_f32_slice().unwrap(), &expected);

    let got = round(&tensor_f64(input.iter().map(|&v| v as f64).collect()), 0).unwrap();
    let want: Vec<f64> = expected.iter().map(|&v| v as f64).collect();
    assert_eq!(got.data().as_f64_slice().unwrap(), want.as_slice());
}

#[test]
fn test_round_to_decimals_also_breaks_ties_to_even() {
    // The `decimals` form scales, rounds and scales back, so the tie rule has to
    // survive that too. 0.125 and 0.375 are exact in binary and both sit on a
    // tie at two decimal places.
    let got = round(&tensor_f32(vec![0.125, 0.375, -0.125, -0.375]), 2).unwrap();
    assert_eq!(
        got.data().as_f32_slice().unwrap(),
        &[0.12f32, 0.38, -0.12, -0.38]
    );
}

#[test]
fn test_round_leaves_non_ties_and_specials_alone() {
    // Changing the tie rule must not disturb anything else.
    let got = round(
        &tensor_f32(vec![0.4, 0.6, -0.4, -0.6, f32::INFINITY, f32::NEG_INFINITY]),
        0,
    )
    .unwrap();
    let values = got.data().as_f32_slice().unwrap();
    assert_eq!(&values[..4], &[0.0f32, 1.0, -0.0, -1.0]);
    assert!(values[4].is_infinite() && values[4].is_sign_positive());
    assert!(values[5].is_infinite() && values[5].is_sign_negative());

    let nan = round(&tensor_f32(vec![f32::NAN]), 0).unwrap();
    assert!(nan.data().as_f32_slice().unwrap()[0].is_nan());
}
