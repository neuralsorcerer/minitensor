// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use approx::assert_relative_eq;
use engine::{
    autograd,
    device::Device,
    operations::{activation, arithmetic, reduction},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use proptest::prelude::*;
use std::sync::Arc;

fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
    let shape_obj = Shape::new(shape);
    let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);
    if let Some(slice) = tensor_data.as_f32_slice_mut() {
        slice.copy_from_slice(&data);
    }
    Tensor::new(
        Arc::new(tensor_data),
        shape_obj,
        DataType::Float32,
        Device::cpu(),
        requires_grad,
    )
}

#[test]
fn test_mul_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
    let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], true);
    let product = arithmetic::mul(&a, &b).unwrap();
    let grad_output = Tensor::ones(
        product.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&product, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[3.0, 4.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[1.0, 2.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_z_leaky_relu_backward_correct() {
    autograd::clear_graph().unwrap();
    let input = create_test_tensor_f32(vec![-1.0, 0.0, 1.0], vec![3], true);
    let output = activation::leaky_relu(&input, 0.1).unwrap();
    let grad_output = Tensor::ones(
        output.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&output, Some(grad_output)).unwrap();
    let grad_input = grads.get(&input.id()).unwrap();
    assert_eq!(grad_input.data().as_f32_slice().unwrap(), &[0.1, 1.0, 1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_relu_backward_nan_propagates() {
    autograd::clear_graph().unwrap();
    let input = create_test_tensor_f32(vec![-1.0, f32::NAN, 1.0], vec![3], true);
    let output = activation::relu(&input).unwrap();
    let grad_output = create_test_tensor_f32(vec![1.0, f32::NAN, 1.0], vec![3], false);
    let grads = autograd::backward(&output, Some(grad_output)).unwrap();
    let grad_input = grads.get(&input.id()).unwrap();
    let vals = grad_input.data().as_f32_slice().unwrap();
    assert_eq!(vals[0], 0.0);
    assert!(vals[1].is_nan());
    assert_eq!(vals[2], 1.0);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_sum_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], true);
    let s = reduction::sum(&a, None, false).unwrap();
    let grad_output = Tensor::ones(s.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&s, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_mean_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let m = reduction::mean(&a, None, false).unwrap();
    let grad_output = Tensor::ones(m.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&m, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(
        grad_a.data().as_f32_slice().unwrap(),
        &[0.25, 0.25, 0.25, 0.25]
    );
    autograd::clear_graph().unwrap();
}

#[test]
#[ignore]
fn test_add_backward_broadcasting() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3, 1], true);
    let b = create_test_tensor_f32(vec![10.0, 20.0], vec![1, 2], true);
    let sum = arithmetic::add(&a, &b).unwrap();
    let grad_output = Tensor::ones(sum.shape().clone(), DataType::Float32, Device::cpu(), false);
    autograd::backward(&sum, Some(grad_output)).unwrap();
    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[2.0, 2.0, 2.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[3.0, 3.0]);
    autograd::clear_graph().unwrap();
}

proptest! {
    #[test]
    fn prop_mul_gradients_are_correct(a_vals in any::<[f32; 2]>(), b_vals in any::<[f32; 2]>()) {
        let a = create_test_tensor_f32(a_vals.to_vec(), vec![2], true);
        let b = create_test_tensor_f32(b_vals.to_vec(), vec![2], true);
        let product = arithmetic::mul(&a, &b).unwrap();
        let grad_output = Tensor::ones(product.shape().clone(), DataType::Float32, Device::cpu(), false);
        let grads = autograd::backward(&product, Some(grad_output)).unwrap();
        let grad_a = grads.get(&a.id()).unwrap();
        let grad_b = grads.get(&b.id()).unwrap();
        let ga = grad_a.data().as_f32_slice().unwrap();
        let gb = grad_b.data().as_f32_slice().unwrap();
        assert_relative_eq!(ga[0], b_vals[0], epsilon = 1e-6);
        assert_relative_eq!(ga[1], b_vals[1], epsilon = 1e-6);
        assert_relative_eq!(gb[0], a_vals[0], epsilon = 1e-6);
        assert_relative_eq!(gb[1], a_vals[1], epsilon = 1e-6);
        autograd::clear_graph().unwrap();
    }
}
