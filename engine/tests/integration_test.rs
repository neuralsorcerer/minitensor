// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    autograd,
    device::Device,
    operations::{activation, arithmetic, comparison, linalg, reduction, shape_ops},
    tensor::{DataType, Shape, Tensor, TensorData},
};
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
fn test_basic_operations_integration() {
    // Create test tensors
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let b = create_test_tensor_f32(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], true);

    // Test arithmetic operations
    let sum = arithmetic::add(&a, &b).unwrap();
    let diff = arithmetic::sub(&sum, &b).unwrap();
    let product = arithmetic::mul(&a, &b).unwrap();
    let _quotient = arithmetic::div(&product, &b).unwrap();

    // Verify arithmetic results
    let sum_data = sum.data().as_f32_slice().unwrap();
    assert_eq!(sum_data, &[3.0, 5.0, 7.0, 9.0]);

    let diff_data = diff.data().as_f32_slice().unwrap();
    assert_eq!(diff_data, &[1.0, 2.0, 3.0, 4.0]); // Should equal original a

    // Test matrix operations
    let transposed = linalg::transpose(&a, 0, 1).unwrap();
    let matmul_result = linalg::matmul(&a, &transposed).unwrap();

    // Verify shapes
    assert_eq!(transposed.shape().dims(), &[2, 2]);
    assert_eq!(matmul_result.shape().dims(), &[2, 2]);

    // Test shape operations
    let reshaped = shape_ops::reshape(&a, Shape::new(vec![4])).unwrap();
    assert_eq!(reshaped.shape().dims(), &[4]);

    let squeezed = shape_ops::squeeze(&reshaped, None).unwrap();
    assert_eq!(squeezed.shape().dims(), &[4]);

    let unsqueezed = shape_ops::unsqueeze(&reshaped, 0).unwrap();
    assert_eq!(unsqueezed.shape().dims(), &[1, 4]);

    // Clear computation graph
    let _ = autograd::clear_graph();
}

#[test]
fn test_comparison_operations() {
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
    let b = create_test_tensor_f32(vec![1.0, 0.0, 4.0], vec![3], false);

    let eq_res = comparison::eq(&a, &b).unwrap();
    let ne_res = comparison::ne(&a, &b).unwrap();
    let lt_res = comparison::lt(&a, &b).unwrap();
    let le_res = comparison::le(&a, &b).unwrap();
    let gt_res = comparison::gt(&a, &b).unwrap();
    let ge_res = comparison::ge(&a, &b).unwrap();

    assert_eq!(
        eq_res.data().as_bool_slice().unwrap(),
        &[true, false, false]
    );
    assert_eq!(ne_res.data().as_bool_slice().unwrap(), &[false, true, true]);
    assert_eq!(
        lt_res.data().as_bool_slice().unwrap(),
        &[false, false, true]
    );
    assert_eq!(le_res.data().as_bool_slice().unwrap(), &[true, false, true]);
    assert_eq!(
        gt_res.data().as_bool_slice().unwrap(),
        &[false, true, false]
    );
    assert_eq!(ge_res.data().as_bool_slice().unwrap(), &[true, true, false]);

    let _ = autograd::clear_graph();
}

#[test]
fn test_activation_functions_integration() {
    // Create test tensor
    let x = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);

    // Test mathematical functions
    let exp_result = activation::exp(&x).unwrap();
    let log_input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], true);
    let log_result = activation::log(&log_input).unwrap();

    // Test trigonometric functions
    let sin_result = activation::sin(&x).unwrap();
    let cos_result = activation::cos(&x).unwrap();
    let tanh_result = activation::tanh(&x).unwrap();

    // Test activation functions
    let sigmoid_result = activation::sigmoid(&x).unwrap();
    let relu_result = activation::relu(&x).unwrap();

    // Test softmax
    let softmax_input = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], true);
    let softmax_result = activation::softmax(&softmax_input, None).unwrap();

    // Verify that all operations maintain gradient tracking
    assert!(exp_result.requires_grad());
    assert!(log_result.requires_grad());
    assert!(sin_result.requires_grad());
    assert!(cos_result.requires_grad());
    assert!(tanh_result.requires_grad());
    assert!(sigmoid_result.requires_grad());
    assert!(relu_result.requires_grad());
    assert!(softmax_result.requires_grad());

    // Verify ReLU behavior
    let relu_data = relu_result.data().as_f32_slice().unwrap();
    assert_eq!(relu_data, &[0.0, 0.0, 0.0, 1.0, 2.0]);

    // Verify softmax sums to 1
    let softmax_data = softmax_result.data().as_f32_slice().unwrap();
    let sum: f32 = softmax_data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Clear computation graph
    let _ = autograd::clear_graph();
}

#[test]
fn test_complex_computation_chain() {
    // Create input tensors
    let x = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let y = create_test_tensor_f32(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], true);

    // Complex computation: sigmoid(tanh(x * y) + relu(x - y))
    let product = arithmetic::mul(&x, &y).unwrap();
    let tanh_product = activation::tanh(&product).unwrap();

    let diff = arithmetic::sub(&x, &y).unwrap();
    let relu_diff = activation::relu(&diff).unwrap();

    let sum = arithmetic::add(&tanh_product, &relu_diff).unwrap();
    let final_result = activation::sigmoid(&sum).unwrap();

    // Verify the computation chain maintains gradient tracking
    assert!(final_result.requires_grad());
    assert!(final_result.grad_fn().is_some());

    // Verify output shape
    assert_eq!(final_result.shape().dims(), &[2, 2]);

    // Verify all values are in sigmoid range [0, 1]
    let result_data = final_result.data().as_f32_slice().unwrap();
    for &val in result_data {
        assert!(val >= 0.0 && val <= 1.0);
    }

    // Clear computation graph
    let _ = autograd::clear_graph();
}

#[test]
fn test_softmax_arbitrary_dimension() {
    // 2x2 tensor
    let input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

    // Softmax along dim 0 (columns)
    let result_dim0 = activation::softmax(&input, Some(0)).unwrap();
    let data0 = result_dim0.data().as_f32_slice().unwrap();
    let max0 = 3.0_f32.max(1.0);
    let e10 = (1.0 - max0).exp();
    let e30 = (3.0 - max0).exp();
    let sum0 = e10 + e30;
    let max1 = 4.0_f32.max(2.0);
    let e21 = (2.0 - max1).exp();
    let e41 = (4.0 - max1).exp();
    let sum1 = e21 + e41;
    let expected0 = [e10 / sum0, e21 / sum1, e30 / sum0, e41 / sum1];
    assert!(
        data0
            .iter()
            .zip(expected0.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );

    // Softmax along dim 1 (rows)
    let result_dim1 = activation::softmax(&input, Some(1)).unwrap();
    let data1 = result_dim1.data().as_f32_slice().unwrap();
    let max_row0 = 2.0_f32.max(1.0);
    let e1 = (1.0 - max_row0).exp();
    let e2 = (2.0 - max_row0).exp();
    let sum_row0 = e1 + e2;
    let max_row1 = 4.0_f32.max(3.0);
    let e3 = (3.0 - max_row1).exp();
    let e4 = (4.0 - max_row1).exp();
    let sum_row1 = e3 + e4;
    let expected1 = [e1 / sum_row0, e2 / sum_row0, e3 / sum_row1, e4 / sum_row1];
    assert!(
        data1
            .iter()
            .zip(expected1.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );

    let _ = autograd::clear_graph();
}

#[test]
fn test_softmax_backward_dim1() {
    let input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let grad_output = create_test_tensor_f32(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2], false);

    let result = activation::softmax(&input, Some(1)).unwrap();
    let grads = autograd::backward(&result, Some(grad_output.clone())).unwrap();
    let grad_data = grads
        .get(&input.id())
        .unwrap()
        .data()
        .as_f32_slice()
        .unwrap();
    let softmax_data = result.data().as_f32_slice().unwrap();
    let grad_out_data = grad_output.data().as_f32_slice().unwrap();

    let mut expected = [0.0f32; 4];
    for i in 0..2 {
        let sum: f32 = (0..2)
            .map(|j| grad_out_data[i * 2 + j] * softmax_data[i * 2 + j])
            .sum();
        for j in 0..2 {
            let idx = i * 2 + j;
            expected[idx] = softmax_data[idx] * (grad_out_data[idx] - sum);
        }
    }

    assert!(
        grad_data
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );

    let _ = autograd::clear_graph();
}

#[test]
fn test_softmax_backward_dim0() {
    let input = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let grad_output = create_test_tensor_f32(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2], false);

    let result = activation::softmax(&input, Some(0)).unwrap();
    let grads = autograd::backward(&result, Some(grad_output.clone())).unwrap();
    let grad_data = grads
        .get(&input.id())
        .unwrap()
        .data()
        .as_f32_slice()
        .unwrap();
    let softmax_data = result.data().as_f32_slice().unwrap();
    let grad_out_data = grad_output.data().as_f32_slice().unwrap();

    let mut expected = [0.0f32; 4];
    for j in 0..2 {
        let sum: f32 = (0..2)
            .map(|i| grad_out_data[i * 2 + j] * softmax_data[i * 2 + j])
            .sum();
        for i in 0..2 {
            let idx = i * 2 + j;
            expected[idx] = softmax_data[idx] * (grad_out_data[idx] - sum);
        }
    }

    assert!(
        grad_data
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );

    let _ = autograd::clear_graph();
}

#[test]
fn test_broadcasting_and_reshaping() {
    // Create tensors with different shapes for broadcasting
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
    let b = create_test_tensor_f32(vec![10.0], vec![1], false);

    // Test broadcasting in addition
    let broadcasted_sum = arithmetic::add(&a, &b).unwrap();
    assert_eq!(broadcasted_sum.shape().dims(), &[3]);

    let sum_data = broadcasted_sum.data().as_f32_slice().unwrap();
    assert_eq!(sum_data, &[11.0, 12.0, 13.0]);

    // Test reshaping
    let matrix = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

    let reshaped = shape_ops::reshape(&matrix, Shape::new(vec![3, 2])).unwrap();
    assert_eq!(reshaped.shape().dims(), &[3, 2]);

    let transposed = linalg::transpose(&reshaped, 0, 1).unwrap();
    assert_eq!(transposed.shape().dims(), &[2, 3]);

    // Clear computation graph
    let _ = autograd::clear_graph();
}

#[test]
fn test_matmul_batch_dimensions() {
    let a = create_test_tensor_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3], false);
    let b = create_test_tensor_f32((0..12).map(|x| x as f32).collect(), vec![2, 3, 2], false);
    let result = linalg::matmul(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2, 2]);
    let expected = [10.0, 13.0, 28.0, 40.0, 172.0, 193.0, 244.0, 274.0];
    assert_eq!(result.data().as_f32_slice().unwrap(), &expected);
    let _ = autograd::clear_graph();
}

#[test]
fn test_matmul_shape_mismatch_error() {
    let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
    let b = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    assert!(linalg::matmul(&a, &b).is_err());
    let _ = autograd::clear_graph();
}

#[test]
fn test_reduction_operations_edges() {
    let t = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);

    let sum_dim1 = reduction::sum(&t, Some(vec![1]), true).unwrap();
    assert_eq!(sum_dim1.shape().dims(), &[2, 1]);
    assert_eq!(sum_dim1.data().as_f32_slice().unwrap(), &[3.0, 7.0]);

    let sum_neg = reduction::sum(&t, Some(vec![-1]), false).unwrap();
    assert_eq!(sum_neg.shape().dims(), &[2]);
    assert_eq!(sum_neg.data().as_f32_slice().unwrap(), &[3.0, 7.0]);

    let mean_all = reduction::mean(&t, None, false).unwrap();
    assert!(mean_all.shape().dims().is_empty());
    assert!((mean_all.data().as_f32_slice().unwrap()[0] - 2.5).abs() < 1e-6);

    let sum_all = reduction::sum(&t, Some(vec![0, 1]), false).unwrap();
    assert!(sum_all.shape().is_scalar());
    assert_eq!(sum_all.data().as_f32_slice().unwrap()[0], 10.0);
    assert!(reduction::sum(&t, Some(vec![2]), false).is_err());
    assert!(reduction::sum(&t, Some(vec![-3]), false).is_err());

    let bool_src = create_test_tensor_f32(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2], false);
    let any_res = reduction::any(&bool_src, Some(1), true).unwrap();
    assert_eq!(any_res.shape().dims(), &[2, 1]);
    assert_eq!(any_res.data().as_bool_slice().unwrap(), &[true, true]);

    let all_res = reduction::all(&bool_src, Some(0), false).unwrap();
    assert_eq!(all_res.shape().dims(), &[2]);
    assert_eq!(all_res.data().as_bool_slice().unwrap(), &[false, false]);

    let _ = autograd::clear_graph();
}
