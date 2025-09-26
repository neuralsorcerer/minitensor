// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use approx::assert_relative_eq;
use engine::{
    autograd,
    device::Device,
    operations::{activation, arithmetic, linalg, reduction},
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
fn test_sub_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![5.0, 7.0], vec![2], true);
    let b = create_test_tensor_f32(vec![3.0, 2.0], vec![2], true);
    let diff = arithmetic::sub(&a, &b).unwrap();
    let grad_output = Tensor::ones(
        diff.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&diff, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[1.0, 1.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[-1.0, -1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_div_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![4.0, 9.0], vec![2], true);
    let b = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
    let quo = arithmetic::div(&a, &b).unwrap();
    let grad_output = Tensor::ones(quo.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&quo, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    let ga = grad_a.data().as_f32_slice().unwrap();
    let gb = grad_b.data().as_f32_slice().unwrap();
    assert_relative_eq!(ga[0], 0.5f32, epsilon = 1e-6);
    assert_relative_eq!(ga[1], 1.0 / 3.0, epsilon = 1e-6);
    assert_relative_eq!(gb[0], -1.0, epsilon = 1e-6); // -4 / 4
    assert_relative_eq!(gb[1], -1.0, epsilon = 1e-6); // -9 / 9
    autograd::clear_graph().unwrap();
}

#[test]
fn test_neg_backward_correct() {
    autograd::clear_graph().unwrap();
    let x = create_test_tensor_f32(vec![1.0, -2.0], vec![2], true);
    let y = arithmetic::neg(&x).unwrap();
    let grad_output = Tensor::ones(y.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&y, Some(grad_output)).unwrap();
    let grad_x = grads.get(&x.id()).unwrap();
    assert_eq!(grad_x.data().as_f32_slice().unwrap(), &[-1.0, -1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_cos_backward_correct() {
    autograd::clear_graph().unwrap();
    let input = create_test_tensor_f32(vec![0.0, std::f32::consts::FRAC_PI_2], vec![2], true);
    let output = activation::cos(&input).unwrap();
    let grad_output = Tensor::ones(
        output.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&output, Some(grad_output)).unwrap();
    let grad_input = grads.get(&input.id()).unwrap();
    let vals = grad_input.data().as_f32_slice().unwrap();
    assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(vals[1], -1.0, epsilon = 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_logsumexp_forward_matches_manual() {
    autograd::clear_graph().unwrap();
    let data = vec![1.0f32, -1.0, 0.5, 2.0, -2.0, 3.0];
    let tensor = create_test_tensor_f32(data.clone(), vec![2, 3], false);
    let reduced = reduction::logsumexp(&tensor, Some(vec![1]), false).unwrap();
    let keepdim = reduction::logsumexp(&tensor, Some(vec![1]), true).unwrap();
    let reduced_vals = reduced.data().as_f32_slice().unwrap();
    let keepdim_vals = keepdim.data().as_f32_slice().unwrap();

    for (row, (&value, chunk)) in reduced_vals.iter().zip(data.chunks(3)).enumerate() {
        let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = chunk.iter().map(|v| (v - max_val).exp()).sum();
        let expected = max_val + sum_exp.ln();
        assert_relative_eq!(value, expected, epsilon = 1e-6);
        assert_relative_eq!(keepdim_vals[row], expected, epsilon = 1e-6);
    }

    // Reduction across all dimensions should match manual computation
    let full = reduction::logsumexp(&tensor, None, false).unwrap();
    let full_val = full.data().as_f32_slice().unwrap()[0];
    let max_all = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_all: f32 = data.iter().map(|v| (v - max_all).exp()).sum();
    let expected_all = max_all + sum_all.ln();
    assert_relative_eq!(full_val, expected_all, epsilon = 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_logsumexp_backward_matches_softmax() {
    autograd::clear_graph().unwrap();
    let input = create_test_tensor_f32(vec![1.0, 1.0, -1.0, 2.0], vec![2, 2], true);
    let output = reduction::logsumexp(&input, Some(vec![1]), false).unwrap();
    let grad_output = Tensor::ones(
        output.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&output, Some(grad_output)).unwrap();
    let grad_input = grads.get(&input.id()).unwrap();
    let grad_vals = grad_input.data().as_f32_slice().unwrap();

    let input_vals = input.data().as_f32_slice().unwrap();
    let cols = input.shape().dims()[1];
    let mut expected = Vec::with_capacity(input_vals.len());
    for row in input_vals.chunks(cols) {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|v| (v - max_val).exp()).sum();
        for &v in row {
            expected.push((v - max_val).exp() / sum_exp);
        }
    }

    for (grad, expected) in grad_vals.iter().zip(expected.iter()) {
        assert_relative_eq!(*grad, *expected, epsilon = 1e-6);
    }
    autograd::clear_graph().unwrap();
}

#[test]
fn test_log_softmax_forward_matches_softmax_log() {
    autograd::clear_graph().unwrap();
    let input = create_test_tensor_f32(vec![1.0, -2.0, 0.5, 4.0, 0.0, -1.0], vec![2, 3], false);

    let log_softmax = activation::log_softmax(&input, Some(1)).unwrap();
    let softmax = activation::softmax(&input, Some(1)).unwrap();

    let log_vals = log_softmax.data().as_f32_slice().unwrap();
    let softmax_vals = softmax.data().as_f32_slice().unwrap();

    for (log_v, soft_v) in log_vals.iter().zip(softmax_vals.iter()) {
        assert_relative_eq!(*log_v, soft_v.ln(), epsilon = 1e-6);
    }

    autograd::clear_graph().unwrap();
}

#[test]
fn test_log_softmax_backward_matches_manual() {
    autograd::clear_graph().unwrap();
    let input_values = vec![1.0, 0.0, -1.0, 2.0, -2.0, 0.5];
    let grad_values = vec![0.2, -0.1, 0.3, -0.4, 0.25, -0.15];
    let input = create_test_tensor_f32(input_values.clone(), vec![2, 3], true);
    let output = activation::log_softmax(&input, Some(1)).unwrap();
    let grad_output = create_test_tensor_f32(grad_values.clone(), vec![2, 3], false);

    let grads = autograd::backward(&output, Some(grad_output)).unwrap();
    let grad_input = grads.get(&input.id()).unwrap();
    let grad_vals = grad_input.data().as_f32_slice().unwrap();

    let mut expected = Vec::with_capacity(grad_vals.len());
    for (log_block, go_block) in output
        .data()
        .as_f32_slice()
        .unwrap()
        .chunks(3)
        .zip(grad_values.chunks(3))
    {
        let sum: f32 = go_block.iter().sum();
        for (&log_v, &go_v) in log_block.iter().zip(go_block.iter()) {
            expected.push(go_v - log_v.exp() * sum);
        }
    }

    for (grad, expected) in grad_vals.iter().zip(expected.iter()) {
        assert_relative_eq!(*grad, *expected, epsilon = 1e-6);
    }

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
fn test_add_backward_broadcasting() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3, 1], true);
    let b = create_test_tensor_f32(vec![10.0, 20.0], vec![1, 2], true);
    let sum = arithmetic::add(&a, &b).unwrap();
    let grad_output = Tensor::ones(sum.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&sum, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[2.0, 2.0, 2.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[3.0, 3.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_mul_backward_broadcasting() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3, 1], true);
    let b = create_test_tensor_f32(vec![10.0, 20.0], vec![1, 2], true);
    let prod = arithmetic::mul(&a, &b).unwrap();
    let grad_output = Tensor::ones(
        prod.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&prod, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[30.0, 30.0, 30.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[6.0, 6.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_sub_backward_broadcasting() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3, 1], true);
    let b = create_test_tensor_f32(vec![10.0, 20.0], vec![1, 2], true);
    let diff = arithmetic::sub(&a, &b).unwrap();
    let grad_output = Tensor::ones(
        diff.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&diff, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[2.0, 2.0, 2.0]);
    assert_eq!(grad_b.data().as_f32_slice().unwrap(), &[-3.0, -3.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_div_backward_broadcasting() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![2.0, 4.0, 6.0], vec![3, 1], true);
    let b = create_test_tensor_f32(vec![2.0, 4.0], vec![1, 2], true);
    let quo = arithmetic::div(&a, &b).unwrap();
    let grad_output = Tensor::ones(quo.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&quo, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    let ga = grad_a.data().as_f32_slice().unwrap();
    let gb = grad_b.data().as_f32_slice().unwrap();
    assert_relative_eq!(ga[0], 0.5 + 0.25, epsilon = 1e-6); // sum over columns 1/2 + 1/4
    assert_relative_eq!(ga[1], 0.5 + 0.25, epsilon = 1e-6);
    assert_relative_eq!(ga[2], 0.5 + 0.25, epsilon = 1e-6);
    assert_relative_eq!(gb[0], -3.0, epsilon = 1e-6);
    assert_relative_eq!(gb[1], -0.75, epsilon = 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_pow_backward_correct() {
    autograd::clear_graph().unwrap();
    let base = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
    let exp = create_test_tensor_f32(vec![3.0, 2.0], vec![2], true);
    let out = activation::pow(&base, &exp).unwrap();
    let grad_output = Tensor::ones(out.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&out, Some(grad_output)).unwrap();
    let grad_base = grads.get(&base.id()).unwrap();
    let grad_exp = grads.get(&exp.id()).unwrap();
    let gb = grad_base.data().as_f32_slice().unwrap();
    let ge = grad_exp.data().as_f32_slice().unwrap();
    assert_relative_eq!(gb[0], 3.0 * 2.0f32.powf(2.0), epsilon = 1e-6);
    assert_relative_eq!(gb[1], 2.0 * 3.0f32.powf(1.0), epsilon = 1e-6);
    assert_relative_eq!(ge[0], 8.0 * 2.0f32.ln(), epsilon = 1e-6);
    assert_relative_eq!(ge[1], 9.0 * 3.0f32.ln(), epsilon = 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_powf_backward_correct() {
    autograd::clear_graph().unwrap();
    let base = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
    let out = activation::powf(&base, 3.0).unwrap();
    let grad_output = Tensor::ones(out.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&out, Some(grad_output)).unwrap();
    let grad_base = grads.get(&base.id()).unwrap();
    let gb = grad_base.data().as_f32_slice().unwrap();
    assert_relative_eq!(gb[0], 3.0 * 2.0f32.powf(2.0), epsilon = 1e-6);
    assert_relative_eq!(gb[1], 3.0 * 3.0f32.powf(2.0), epsilon = 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_broadcast_backward_multiple_axes() {
    autograd::clear_graph().unwrap();
    // b is broadcast across both axes
    let a = create_test_tensor_f32(vec![1.0; 6], vec![2, 3], true);
    let b = create_test_tensor_f32(vec![2.0], vec![1, 1], true);
    let sum = arithmetic::add(&a, &b).unwrap();
    let grad_output = Tensor::ones(sum.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&sum, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    let grad_b = grads.get(&b.id()).unwrap();
    assert_eq!(grad_a.shape().dims(), &[2, 3]);
    assert_eq!(grad_b.shape().dims(), &[1, 1]);
    let slice = grad_b.data().as_f32_slice().unwrap();
    assert!((slice[0] - 6.0).abs() < 1e-6);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_sum_backward_with_dims_keepdim() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let s = reduction::sum(&a, Some(vec![1]), false).unwrap();
    let grad_output = Tensor::ones(s.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&s, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
    autograd::clear_graph().unwrap();

    let s_keep = reduction::sum(&a, Some(vec![1]), true).unwrap();
    let grad_output = Tensor::ones(
        s_keep.shape().clone(),
        DataType::Float32,
        Device::cpu(),
        false,
    );
    let grads = autograd::backward(&s_keep, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_transpose_backward_correct() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let t = linalg::transpose(&a, 0, 1).unwrap();
    let grad_output = Tensor::ones(t.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&t, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
    autograd::clear_graph().unwrap();
}

#[test]
fn test_gradient_accumulation_multiple_paths() {
    autograd::clear_graph().unwrap();
    let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
    let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], true);
    let c = create_test_tensor_f32(vec![5.0, 6.0], vec![2], true);
    let temp1 = arithmetic::add(&a, &b).unwrap();
    let temp2 = arithmetic::add(&a, &c).unwrap();
    let d = arithmetic::add(&temp1, &temp2).unwrap();
    let grad_output = Tensor::ones(d.shape().clone(), DataType::Float32, Device::cpu(), false);
    let grads = autograd::backward(&d, Some(grad_output)).unwrap();
    let grad_a = grads.get(&a.id()).unwrap();
    assert_eq!(grad_a.data().as_f32_slice().unwrap(), &[2.0, 2.0]);
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
