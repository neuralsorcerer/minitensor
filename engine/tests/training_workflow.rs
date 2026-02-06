// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    autograd,
    operations::{arithmetic, linalg, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

fn create_tensor(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
    let shape_obj = Shape::new(shape);
    let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);
    if let Some(slice) = tensor_data.as_f32_slice_mut() {
        slice.copy_from_slice(&data);
    }
    Tensor::new(
        Arc::new(tensor_data),
        shape_obj,
        DataType::Float32,
        engine::device::Device::cpu(),
        requires_grad,
    )
}

#[test]
#[ignore]
fn test_linear_regression_training() {
    let x = create_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1], false);
    let y_true = create_tensor(vec![5.0, 7.0, 9.0, 11.0], vec![4, 1], false); // y = 2x + 3
    let mut w = create_tensor(vec![0.0], vec![1, 1], true);
    let mut b = create_tensor(vec![0.0], vec![1], true);
    let lr = 0.05f32;

    for _ in 0..5000 {
        let y_pred = arithmetic::add(&linalg::matmul(&x, &w).unwrap(), &b).unwrap();
        let diff = arithmetic::sub(&y_pred, &y_true).unwrap();
        let sq = arithmetic::mul(&diff, &diff).unwrap();
        let loss = reduction::mean(&sq, None, false).unwrap();
        autograd::backward(&loss, None).unwrap();

        let grad_w = autograd::get_gradient(&w).unwrap().detach();
        let grad_b = autograd::get_gradient(&b).unwrap().detach();
        let lr_tensor = create_tensor(vec![lr], vec![1], false);
        let step_w = arithmetic::mul(&grad_w, &lr_tensor).unwrap();
        let step_b = arithmetic::mul(&grad_b, &lr_tensor).unwrap();
        w = arithmetic::sub(&w, &step_w)
            .unwrap()
            .detach()
            .requires_grad_(true);
        b = arithmetic::sub(&b, &step_b)
            .unwrap()
            .detach()
            .requires_grad_(true);
        autograd::clear_graph().unwrap();
    }

    let w_val = w.data().as_f32_slice().unwrap()[0];
    let b_val = b.data().as_f32_slice().unwrap()[0];
    assert!((w_val - 2.0).abs() < 0.1);
    assert!((b_val - 3.0).abs() < 0.1);
}
