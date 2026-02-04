// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn ceil_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::ceil);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        autograd,
        device::Device,
        tensor::{Shape, Tensor, TensorData},
    };

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

    fn create_test_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let tensor_data = TensorData::from_vec_bool(data, Device::cpu());
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_exp() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, 2.0], vec![3], false);
        let result = exp(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!((result_data[1] - std::f32::consts::E).abs() < 1e-6);
        assert!((result_data[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 1e-5);
    }

    #[test]
    fn test_exp_invalid_dtype() {
        let shape = Shape::new(vec![3]);
        let data = TensorData::from_vec_i32(vec![1, 2, 3], Device::cpu());
        let tensor = Tensor::new(Arc::new(data), shape, DataType::Int32, Device::cpu(), false);
        assert!(exp(&tensor).is_err());
    }

    #[test]
    fn test_log() {
        let tensor = create_test_tensor_f32(
            vec![
                1.0,
                std::f32::consts::E,
                std::f32::consts::E * std::f32::consts::E,
            ],
            vec![3],
            false,
        );
        let result = log(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!((result_data[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sin() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = sin(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0).abs() < 1e-6);
        assert!(result_data[2].abs() < 1e-6); // sin(π) ≈ 0
    }

    #[test]
    fn test_cos() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
            vec![3],
            false,
        );
        let result = cos(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 1.0).abs() < 1e-6);
        assert!(result_data[1].abs() < 1e-6); // cos(π/2) ≈ 0
        assert!((result_data[2] + 1.0).abs() < 1e-6); // cos(π) ≈ -1
    }

    #[test]
    fn test_tan() {
        let tensor = create_test_tensor_f32(
            vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0],
            vec![3],
            false,
        );
        let result = tan(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - (std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
        assert!((result_data[2] - (-std::f32::consts::PI / 4.0).tan()).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = tanh(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 1.0_f32.tanh()).abs() < 1e-6);
        assert!((result_data[2] - (-1.0_f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let tensor = create_test_tensor_f32(vec![0.0, 1.0, -1.0], vec![3], false);
        let result = sigmoid(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!((result_data[0] - 0.5).abs() < 1e-6);
        assert!((result_data[1] - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
        assert!((result_data[2] - (1.0 / (1.0 + 1.0_f32.exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = relu(&tensor).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_hardshrink_forward() {
        let tensor = create_test_tensor_f32(vec![-1.2, -0.2, 0.0, 0.45, 0.9], vec![5], false);
        let result = hardshrink(&tensor, 0.3).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[-1.2, 0.0, 0.0, 0.45, 0.9]);
    }

    #[test]
    fn test_hardshrink_backward() {
        let tensor = create_test_tensor_f32(vec![-1.2, -0.25, 0.0, 0.35, 0.8], vec![5], true);
        let result = hardshrink(&tensor, 0.3).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let grad_vals = grad.data().as_f32_slice().unwrap();
        assert_eq!(grad_vals, &[1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_hardshrink_invalid_lambda() {
        let tensor = create_test_tensor_f32(vec![1.0], vec![1], false);
        assert!(hardshrink(&tensor, -0.1).is_err());
    }

    #[test]
    fn test_leaky_relu() {
        let tensor = create_test_tensor_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
        let result = leaky_relu(&tensor, 0.1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = softmax(&tensor, None).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in result_data {
            assert!(val > 0.0);
        }

        // Check that larger input values produce larger probabilities
        assert!(result_data[2] > result_data[1]);
        assert!(result_data[1] > result_data[0]);
    }

    #[test]
    fn test_masked_softmax_respects_mask() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let mask = create_test_tensor_bool(vec![true, false, false, true], vec![2, 2]);
        let result = masked_softmax(&tensor, &mask, Some(1)).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[3], 0.0);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_masked_softmax_all_masked_returns_zero() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let mask = create_test_tensor_bool(vec![true, true], vec![2]);
        let result = masked_softmax(&tensor, &mask, Some(0)).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_masked_softmax_all_negative_infinity_unmasked() {
        let tensor =
            create_test_tensor_f32(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], vec![2], false);
        let mask = create_test_tensor_bool(vec![false, false], vec![2]);
        let result = masked_softmax(&tensor, &mask, Some(0)).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_masked_log_softmax_broadcast_mask() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let mask = create_test_tensor_bool(vec![true, false], vec![2, 1]);
        let result = masked_log_softmax(&tensor, &mask, Some(1)).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!(data[0].is_infinite() && data[0].is_sign_negative());
        assert!(data[1].is_infinite() && data[1].is_sign_negative());
        let row1_prob0 = data[2].exp();
        let row1_prob1 = data[3].exp();
        assert!((row1_prob0 + row1_prob1 - 1.0).abs() < 1e-6);
        assert!(data[3] > data[2]);
    }

    #[test]
    fn test_log_softmax_large_negative_values() {
        let tensor = create_test_tensor_f32(vec![-1000.0, 0.0], vec![2], false);
        let result = log_softmax(&tensor, None).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!(result_data[0].is_infinite() && result_data[0].is_sign_negative());
        assert!(result_data[1].abs() < 1e-6);
    }

    #[test]
    fn test_log_softmax_all_negative_infinity() {
        let tensor =
            create_test_tensor_f32(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], vec![2], false);
        let result = log_softmax(&tensor, None).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert!(
            result_data
                .iter()
                .all(|v| v.is_infinite() && v.is_sign_negative())
        );
    }

    #[test]
    fn test_powf_scalar() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = powf(&tensor, 2.0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_pow_tensor() {
        let base = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let exp = create_test_tensor_f32(vec![1.0, 2.0, 0.5], vec![3], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_shape_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let exp = create_test_tensor_f32(vec![3.0, 4.0, 5.0], vec![3], false);
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_pow_dtype_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let shape = Shape::new(vec![2]);
        let data = TensorData::from_vec_f64(vec![1.0, 2.0], Device::cpu());
        let exp = Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float64,
            Device::cpu(),
            false,
        );
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_pow_device_mismatch_error() {
        let base = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let shape = Shape::new(vec![2]);
        let data = TensorData::from_vec_f32(vec![1.0, 2.0], Device::cuda(Some(0)));
        let exp = Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cuda(Some(0)),
            false,
        );
        assert!(pow(&base, &exp).is_err());
    }

    #[test]
    fn test_powf_gradient() {
        let tensor = create_test_tensor_f32(vec![2.0, 3.0], vec![2], true);
        let result = powf(&tensor, 3.0).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - 3.0 * 2.0_f32.powf(2.0)).abs() < 1e-6);
        assert!((g[1] - 3.0 * 3.0_f32.powf(2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pow_base_scalar_tensor_exponent() {
        let base = create_test_tensor_f32(vec![2.0], vec![], false);
        let exp = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_exponent_scalar_tensor_base() {
        let base = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let exp = create_test_tensor_f32(vec![2.0], vec![1], false);
        let result = pow(&base, &exp).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
        assert!((data[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_pow_base_scalar_gradient() {
        let base = create_test_tensor_f32(vec![2.0], vec![], true);
        let exp = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let result = pow(&base, &exp).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&base.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        let base_val = base.data().as_f32_slice().unwrap()[0];
        let exp_vals = exp.data().as_f32_slice().unwrap();
        let expected = exp_vals
            .iter()
            .map(|&e| e * base_val.powf(e - 1.0))
            .sum::<f32>();
        assert!((g[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_pow_exponent_scalar_gradient() {
        let base = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);
        let exp = create_test_tensor_f32(vec![1.5], vec![1], true);
        let result = pow(&base, &exp).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&exp.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        let exp_val = exp.data().as_f32_slice().unwrap()[0];
        let base_vals = base.data().as_f32_slice().unwrap();
        let expected = base_vals
            .iter()
            .map(|&b| b.powf(exp_val) * b.ln())
            .sum::<f32>();
        assert!((g[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt() {
        let tensor = create_test_tensor_f32(vec![1.0, 4.0, 9.0], vec![3], false);
        let result = sqrt(&tensor).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sqrt_gradient() {
        let tensor = create_test_tensor_f32(vec![4.0, 9.0], vec![2], true);
        let result = sqrt(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - 0.25).abs() < 1e-6);
        assert!((g[1] - (1.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rsqrt() {
        let tensor = create_test_tensor_f32(vec![0.25, 1.0, 4.0], vec![3], false);
        let result = rsqrt(&tensor).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rsqrt_gradient() {
        let tensor = create_test_tensor_f32(vec![0.25, 4.0], vec![2], true);
        let result = rsqrt(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad = grads.get(&tensor.id()).unwrap();
        let g = grad.data().as_f32_slice().unwrap();
        assert!((g[0] - (-0.5 * 0.25_f32.powf(-1.5))).abs() < 1e-5);
        assert!((g[1] - (-0.5 * 4.0_f32.powf(-1.5))).abs() < 1e-6);
    }

    #[test]
    fn test_softsign_forward() {
        let data = vec![-2.5f32, -0.5, 0.0, 0.25, 4.0];
        let tensor = create_test_tensor_f32(data.clone(), vec![5], false);

        let result = softsign(&tensor).unwrap();
        let values = result.data().as_f32_slice().unwrap();

        for (out, &x) in values.iter().zip(data.iter()) {
            let denom = 1.0 + x.abs();
            let expected = x / denom;
            assert!((out - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softsign_gradient() {
        let data = vec![-1.5f32, -0.25, 0.5, 3.0];
        let tensor = create_test_tensor_f32(data.clone(), vec![4], true);

        let result = softsign(&tensor).unwrap();
        let ones = Tensor::ones(
            result.shape().clone(),
            result.dtype(),
            result.device(),
            false,
        );
        let grads = autograd::backward(&result, Some(ones)).unwrap();
        let grad_tensor = grads.get(&tensor.id()).unwrap();
        let grad_data = grad_tensor.data().as_f32_slice().unwrap();

        for ((&grad, &x), idx) in grad_data.iter().zip(data.iter()).zip(0..) {
            let denom = 1.0 + x.abs();
            let expected = 1.0 / (denom * denom);
            assert!(
                (grad - expected).abs() < 1e-5,
                "gradient mismatch at index {idx}: got {grad}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gradient_tracking() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], true);

        let result = relu(&tensor).unwrap();
        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());

        let result2 = sigmoid(&tensor).unwrap();
        assert!(result2.requires_grad());
        assert!(result2.grad_fn().is_some());
    }
}
