// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

/// Generic transpose implementation
fn transpose_generic<T: Copy + Send + Sync>(
    input_data: &[T],
    output_data: &mut [T],
    input_shape: &Shape,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    // Fast path for 2D matrix transpose
    if input_shape.ndim() == 2 && dim0 == 0 && dim1 == 1 {
        let rows = input_shape.dims()[0];
        let cols = input_shape.dims()[1];
        if rows * cols < PAR_THRESHOLD {
            for i in 0..rows {
                for j in 0..cols {
                    unsafe {
                        *output_data.get_unchecked_mut(j * rows + i) =
                            *input_data.get_unchecked(i * cols + j);
                    }
                }
            }
        } else {
            output_data
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(j, col)| {
                    for i in 0..rows {
                        unsafe {
                            col[i] = *input_data.get_unchecked(i * cols + j);
                        }
                    }
                });
        }
        return Ok(());
    }

    let input_strides = Strides::from_shape(input_shape);
    let output_strides = Strides::from_shape(output_shape);
    let in_strides = input_strides.as_slice().to_vec();
    let out_strides = output_strides.as_slice().to_vec();
    let out_dims = output_shape.dims().to_vec();

    output_data
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, out)| {
            let mut remaining = idx;
            let mut input_linear = 0;
            for dim in 0..out_dims.len() {
                let stride = out_strides[dim];
                let coord = remaining / stride;
                remaining %= stride;
                let in_dim = if dim == dim0 {
                    dim1
                } else if dim == dim1 {
                    dim0
                } else {
                    dim
                };
                input_linear += coord * in_strides[in_dim];
            }
            *out = input_data[input_linear];
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autograd::GradientFunction, device::Device, tensor::TensorData};

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

    fn create_test_tensor_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float64);

        if let Some(slice) = tensor_data.as_f64_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        )
    }

    fn create_test_tensor_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);

        if let Some(slice) = tensor_data.as_i32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn create_test_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Bool);

        if let Some(slice) = tensor_data.as_bool_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    fn create_test_tensor_f32_on_device(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data =
            TensorData::zeros_on_device(shape_obj.numel(), DataType::Float32, device);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            device,
            false,
        )
    }

    #[test]
    fn test_matmul_basic() {
        // 2x3 * 3x2 = 2x2
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let b = create_test_tensor_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false);

        let result = matmul(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64; 139, 154]
        assert_eq!(result_data, &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_matmul_i32_zero_k_dimension() {
        let a = create_test_tensor_i32(vec![], vec![2, 0]);
        let b = create_test_tensor_i32(vec![], vec![0, 3]);

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
        assert_eq!(result.data().as_i32_slice().unwrap(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = transpose(&a, 0, 1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result_data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0, 5.0], vec![3, 1], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_same_dim() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = transpose(&a, 0, 0).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Should be unchanged
        assert_eq!(result_data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_gradient_tracking() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], true);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2, 1], true);

        let result = matmul(&a, &b).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
    }

    #[test]
    fn test_matmul_dtype_mismatch() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_device_mismatch() {
        let a =
            create_test_tensor_f32_on_device(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::cpu());
        let b = create_test_tensor_f32_on_device(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            Device::cuda(None),
        );

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_bool_error() {
        let a = create_test_tensor_bool(vec![true, false, true, false], vec![2, 2]);
        let b = create_test_tensor_bool(vec![true, true, false, false], vec![2, 2]);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_requires_2d_inputs() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_bmm_basic() {
        let a = create_test_tensor_f32(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
            ],
            vec![2, 2, 3],
            false,
        );
        let b = create_test_tensor_f32(
            vec![
                0.5, 1.0, 1.5, 2.0, 2.5, 3.0, // batch 0
                3.5, 4.0, 4.5, 5.0, 5.5, 6.0, // batch 1
            ],
            vec![2, 3, 2],
            false,
        );

        let result = bmm(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        assert_eq!(
            result_data,
            &[11.0, 14.0, 24.5, 32.0, 110.0, 122.0, 150.5, 167.0]
        );
    }

    #[test]
    fn test_bmm_batch_mismatch() {
        let a = create_test_tensor_f32(vec![1.0; 12], vec![2, 2, 3], false);
        let b = create_test_tensor_f32(vec![2.0; 18], vec![3, 3, 2], false);

        let result = bmm(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_bmm_rank_error() {
        let a = create_test_tensor_f32(vec![1.0; 6], vec![2, 3], false);
        let b = create_test_tensor_f32(vec![2.0; 6], vec![1, 3, 2], false);

        let result = bmm(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_diagonal_main() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = diagonal(&tensor, 0, 0, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2]);
    }

    #[test]
    fn test_diagonal_with_offset() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let upper = diagonal(&tensor, 1, 0, 1).unwrap();
        assert_eq!(upper.data().as_f32_slice().unwrap(), &[2.0, 6.0]);

        let lower = diagonal(&tensor, -1, 0, 1).unwrap();
        assert_eq!(lower.data().as_f32_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_diagonal_high_dim_shape() {
        let tensor =
            create_test_tensor_f32((0..24).map(|v| v as f32).collect(), vec![2, 3, 4], false);
        let result = diagonal(&tensor, 0, 1, 2).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_diagonal_backward_gradients() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let grad_output = create_test_tensor_f32(vec![1.0, 1.0], vec![2], false);

        let backward_fn = crate::autograd::DiagonalBackward {
            input_shape: tensor.shape().dims().to_vec(),
            input_strides: tensor.strides().as_slice().to_vec(),
            input_dtype: DataType::Float32,
            dim1: 0,
            dim2: 1,
            offset: 0,
            input_requires_grad: true,
            input_id: tensor.id(),
        };

        let gradients = backward_fn.backward(&grad_output).unwrap();
        let grad_tensor = gradients.get(&tensor.id()).unwrap();
        let grad = grad_tensor.data().as_f32_slice().unwrap();
        assert_eq!(grad, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_trace_matches_manual_sum() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let traced = trace(&tensor, 0, 0, 1).unwrap();
        let value = traced.data().as_f32_slice().unwrap();
        assert_eq!(value, &[5.0]);
    }

    #[test]
    fn test_triu_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_triu_with_positive_diagonal() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tril_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tril_with_negative_diagonal() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, -1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 0.0, 3.0, 0.0]);
    }
}
