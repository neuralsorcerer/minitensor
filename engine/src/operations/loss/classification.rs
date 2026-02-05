// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn fill_one_hot_f64<T, F>(
    indices: &[T],
    out: &mut [f64],
    num_classes: usize,
    to_index: F,
) -> Result<()>
where
    F: Fn(&T) -> Result<usize>,
{
    for (i, value) in indices.iter().enumerate() {
        let class = to_index(value)?;
        out[i * num_classes + class] = 1.0;
    }
    Ok(())
}

/// Compute the sign of each tensor element (-1.0, 0.0, or 1.0)
fn sign(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let v = *in_ptr.add(i);
                        *out_ptr.add(i) = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                });
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let v = *in_ptr.add(i);
                        *out_ptr.add(i) = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sign operation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Sum all elements in a tensor to produce a scalar
fn sum_all_elements(tensor: &Tensor) -> Result<Tensor> {
    let scalar_shape = Shape::new(vec![1]);
    let mut output_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let sum: f32 = input_data
                .par_chunks(CHUNK)
                .map(|chunk| unsafe {
                    let mut acc = 0f32;
                    let ptr = chunk.as_ptr();
                    for i in 0..chunk.len() {
                        acc += *ptr.add(i);
                    }
                    acc
                })
                .sum();
            output_slice[0] = sum;
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            let sum: f64 = input_data
                .par_chunks(CHUNK)
                .map(|chunk| unsafe {
                    let mut acc = 0f64;
                    let ptr = chunk.as_ptr();
                    for i in 0..chunk.len() {
                        acc += *ptr.add(i);
                    }
                    acc
                })
                .sum();
            output_slice[0] = sum;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Sum only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        scalar_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Divide tensor by a scalar value
fn divide_by_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let scalar_f32 = scalar as f32;
            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        *out_ptr.add(i) = *in_ptr.add(i) / scalar_f32;
                    }
                });
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(input_data.par_chunks(CHUNK))
                .for_each(|(out, inp)| unsafe {
                    let in_ptr = inp.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        *out_ptr.add(i) = *in_ptr.add(i) / scalar;
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Division only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Create a scalar tensor with the given value
fn create_scalar_tensor(
    value: f64,
    dtype: DataType,
    device: crate::device::Device,
) -> Result<Tensor> {
    let scalar_shape = Shape::new(vec![1]);
    let mut tensor_data = TensorData::zeros_on_device(1, dtype, device);

    match dtype {
        DataType::Float32 => {
            let slice = tensor_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = tensor_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensor creation only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        scalar_shape,
        dtype,
        device,
        false,
    ))
}

/// Compute Huber loss element-wise
fn compute_huber_elementwise(
    abs_diff: &Tensor,
    diff: &Tensor,
    _delta_tensor: &Tensor,
    delta: f64,
) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(abs_diff.numel(), abs_diff.dtype(), abs_diff.device());

    match abs_diff.dtype() {
        DataType::Float32 => {
            let abs_data = abs_diff.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from abs_diff")
            })?;
            let diff_data = diff.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from diff")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let delta_f32 = delta as f32;
            output_slice
                .par_chunks_mut(CHUNK)
                .zip(abs_data.par_chunks(CHUNK).zip(diff_data.par_chunks(CHUNK)))
                .for_each(|(out, (abs_chunk, diff_chunk))| unsafe {
                    let abs_ptr = abs_chunk.as_ptr();
                    let diff_ptr = diff_chunk.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let abs_val = *abs_ptr.add(i);
                        *out_ptr.add(i) = if abs_val <= delta_f32 {
                            0.5 * *diff_ptr.add(i) * *diff_ptr.add(i)
                        } else {
                            delta_f32 * (abs_val - 0.5 * delta_f32)
                        };
                    }
                });
        }
        DataType::Float64 => {
            let abs_data = abs_diff.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from abs_diff")
            })?;
            let diff_data = diff.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from diff")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            output_slice
                .par_chunks_mut(CHUNK)
                .zip(abs_data.par_chunks(CHUNK).zip(diff_data.par_chunks(CHUNK)))
                .for_each(|(out, (abs_chunk, diff_chunk))| unsafe {
                    let abs_ptr = abs_chunk.as_ptr();
                    let diff_ptr = diff_chunk.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    for i in 0..out.len() {
                        let abs_val = *abs_ptr.add(i);
                        *out_ptr.add(i) = if abs_val <= delta {
                            0.5 * *diff_ptr.add(i) * *diff_ptr.add(i)
                        } else {
                            delta * (abs_val - 0.5 * delta)
                        };
                    }
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Huber loss only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        abs_diff.shape().clone(),
        abs_diff.dtype(),
        abs_diff.device(),
        abs_diff.requires_grad(),
    ))
}

/// Compute natural logarithm of tensor elements
fn log(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                if val <= 0.0 {
                    output_slice[i] = f32::NEG_INFINITY;
                } else {
                    output_slice[i] = val.ln();
                }
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                if val <= 0.0 {
                    output_slice[i] = f64::NEG_INFINITY;
                } else {
                    output_slice[i] = val.ln();
                }
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Logarithm only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Negate tensor elements
fn negate(tensor: &Tensor) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = -val;
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = -val;
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Negation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// Compute negative log likelihood for classification
fn negative_log_likelihood(log_predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Simplified implementation - multiply log predictions by targets and negate
    let likelihood = mul(log_predictions, targets)?;
    negate(&likelihood)
}

/// Raise tensor elements to a power
fn power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;

            let exp_f32 = exponent as f32;
            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.powf(exp_f32);
            }
        }
        DataType::Float64 => {
            let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;

            for (i, &val) in input_data.iter().enumerate() {
                output_slice[i] = val.powf(exponent);
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

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
    fn test_mse_loss_mean() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.5], vec![3], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
        assert_eq!(loss.shape().dims(), &[1]);
    }

    #[test]
    fn test_mse_loss_sum() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "sum").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (1.0-2.0)² + (2.0-3.0)² = 1.0 + 1.0 = 2.0
        assert!((loss_data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_none() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![2.0, 3.0], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [(1.0-2.0)², (2.0-3.0)²] = [1.0, 1.0]
        assert!((loss_data[0] - 1.0).abs() < 1e-6);
        assert!((loss_data[1] - 1.0).abs() < 1e-6);
        assert_eq!(loss.shape().dims(), &[2]);
    }

    #[test]
    fn test_mae_loss_mean() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 2.0], vec![3], false);

        let loss = mae_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: (|1.0-1.5| + |2.0-2.5| + |3.0-2.0|) / 3 = (0.5 + 0.5 + 1.0) / 3 = 2.0/3 ≈ 0.667
        assert!((loss_data[0] - (2.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.2, 2.3], vec![2], false);

        // Delta = 1.0, differences are 0.2 and 0.3, both <= 1.0, so quadratic
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [0.5 * 0.2², 0.5 * 0.3²] = [0.02, 0.045]
        assert!((loss_data[0] - 0.02).abs() < 1e-6);
        assert!((loss_data[1] - 0.045).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![3.0, 0.0], vec![2], false);

        // Delta = 1.0, differences are 2.0 and 2.0, both > 1.0, so linear
        let loss = huber_loss(&predictions, &targets, 1.0, "none").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        // Expected: [1.0 * (2.0 - 0.5 * 1.0), 1.0 * (2.0 - 0.5 * 1.0)] = [1.5, 1.5]
        assert!((loss_data[0] - 1.5).abs() < 1e-6);
        assert!((loss_data[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_mean_and_backward() {
        let predictions = create_test_tensor_f32(vec![0.8, 0.2], vec![2], true);
        let targets = create_test_tensor_f32(vec![1.0, 0.0], vec![2], false);

        let loss = binary_cross_entropy_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        let expected = -((0.8f32).ln() + (0.8f32).ln()) / 2.0;
        assert!((loss_val - expected).abs() < 1e-6);

        let grads = crate::autograd::backward(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(1.0 / 0.8) / 2.0, (1.0 / 0.8) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    #[test]
    fn test_kl_div_loss_mean_and_backward() {
        let predictions = create_test_tensor_f32(vec![0.4, 0.6], vec![2], true);
        let targets = create_test_tensor_f32(vec![0.5, 0.5], vec![2], false);

        let loss = kl_div_loss(&predictions, &targets, "mean").unwrap();
        let loss_val = loss.data().as_f32_slice().unwrap()[0];
        let expected = 0.5 * ((0.5f32.ln() - 0.4f32.ln()) + (0.5f32.ln() - 0.6f32.ln()));
        assert!((loss_val - expected).abs() < 1e-6);

        let grads = crate::autograd::backward(&loss, None).unwrap();
        let grad = grads.get(&predictions.id()).unwrap();
        let grad_slice = grad.data().as_f32_slice().unwrap();
        let expected_grad = [-(0.5 / 0.4) / 2.0, -(0.5 / 0.6) / 2.0];
        assert!((grad_slice[0] - expected_grad[0]).abs() < 1e-6);
        assert!((grad_slice[1] - expected_grad[1]).abs() < 1e-6);
    }

    #[test]
    fn test_loss_gradient_tracking() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let loss = mse_loss(&predictions, &targets, "mean").unwrap();

        assert!(loss.requires_grad());
        assert!(loss.grad_fn().is_some());
    }

    #[test]
    fn test_loss_input_validation() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5, 3.5], vec![3], false);

        // Shape mismatch should fail
        let result = mse_loss(&predictions, &targets, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_reduction_mode() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let result = mse_loss(&predictions, &targets, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_huber_loss_invalid_delta() {
        let predictions = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![1.5, 2.5], vec![2], false);

        let result = huber_loss(&predictions, &targets, -1.0, "mean");
        assert!(result.is_err());
    }

    #[test]
    fn test_smooth_l1_loss_matches_huber() {
        let predictions = create_test_tensor_f32(vec![0.5, 2.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![0.0, 0.0], vec![2], false);

        let smooth = smooth_l1_loss(&predictions, &targets, "none").unwrap();
        let huber = huber_loss(&predictions, &targets, 1.0, "none").unwrap();

        let smooth_data = smooth.data().as_f32_slice().unwrap();
        let huber_data = huber.data().as_f32_slice().unwrap();
        assert!((smooth_data[0] - huber_data[0]).abs() < 1e-6);
        assert!((smooth_data[1] - huber_data[1]).abs() < 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_mean() {
        let predictions = create_test_tensor_f32(vec![0.0, 1.0], vec![2], false);
        let targets = create_test_tensor_f32(vec![0.0, 0.0], vec![2], false);

        let loss = log_cosh_loss(&predictions, &targets, "mean").unwrap();
        let loss_data = loss.data().as_f32_slice().unwrap();

        let expected = (0.0f32.cosh().ln() + 1.0f32.cosh().ln()) / 2.0;
        assert!((loss_data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_invalid_reduction() {
        let predictions = create_test_tensor_f32(vec![0.0], vec![1], false);
        let targets = create_test_tensor_f32(vec![0.0], vec![1], false);

        let result = log_cosh_loss(&predictions, &targets, "invalid");
        assert!(result.is_err());
    }
}
