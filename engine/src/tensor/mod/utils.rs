// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("tensor_id", &self.tensor_id)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .field("has_grad", &self.grad.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::data::TensorData;

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new(vec![2, 3]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), false);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), DataType::Float32);
        assert_eq!(tensor.device(), Device::cpu());
        assert!(!tensor.requires_grad());
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_tensor_view() {
        let shape = Shape::new(vec![2, 3]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape, DataType::Float32, Device::cpu(), false);

        let new_shape = Shape::new(vec![3, 2]);
        let reshaped = tensor.view(new_shape.clone()).unwrap();
        assert_eq!(reshaped.shape(), &new_shape);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_tensor_zeros_and_ones() {
        let shape = Shape::new(vec![2, 3]);

        let zeros = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), false);
        assert_eq!(zeros.shape(), &shape);
        assert_eq!(zeros.dtype(), DataType::Float32);
        assert!(!zeros.requires_grad());

        let ones = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), true);
        assert_eq!(ones.shape(), &shape);
        assert_eq!(ones.dtype(), DataType::Float32);
        assert!(ones.requires_grad());
    }

    #[test]
    fn test_gradient_management() {
        let shape = Shape::new(vec![2, 2]);
        let mut tensor = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        // Initially no gradient
        assert!(!tensor.has_grad());
        assert!(tensor.grad().is_none());

        // Set a gradient
        let grad = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        assert!(tensor.has_grad());
        assert!(tensor.grad().is_some());

        // Clear gradient (should zero it in place)
        tensor.zero_grad(false);
        assert!(tensor.has_grad());
        let expected = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), false);
        assert!(tensor.grad().unwrap().allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_gradient_accumulation() {
        let shape = Shape::new(vec![2, 2]);
        let mut tensor = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        let grad1 = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);
        let grad2 = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);

        // Accumulate first gradient
        tensor.accumulate_grad(grad1).unwrap();
        assert!(tensor.has_grad());

        // Accumulate second gradient (should replace for now)
        tensor.accumulate_grad(grad2).unwrap();
        assert!(tensor.has_grad());
    }

    #[test]
    fn test_backward_scalar_tensor() {
        let shape = Shape::new(vec![1]);
        let tensor = Tensor::ones(shape, DataType::Float32, Device::cpu(), true);

        // This should work for scalar tensors and produce a gradient
        let result = tensor.backward(None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_non_scalar_error() {
        let tensor = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let result = tensor.backward(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_isnan_isinf_isfinite() {
        let data = vec![0.0f32, f32::NAN, f32::INFINITY, -5.0];
        let shape = Shape::new(vec![4]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let isnan = tensor.isnan().unwrap();
        let isinf = tensor.isinf().unwrap();
        let isfinite = tensor.isfinite().unwrap();

        let isnan_data = isnan.data().as_bool_slice().unwrap();
        let isinf_data = isinf.data().as_bool_slice().unwrap();
        let isfinite_data = isfinite.data().as_bool_slice().unwrap();

        assert_eq!(isnan_data, &[false, true, false, false]);
        assert_eq!(isinf_data, &[false, false, true, false]);
        assert_eq!(isfinite_data, &[true, false, false, true]);
        assert_eq!(isnan.shape(), &shape);
    }

    #[test]
    fn test_clamp() {
        let data = vec![-2.0f32, -0.5, 0.5, 2.0];
        let shape = Shape::new(vec![4]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let clamped = tensor.clamp(Some(-1.0), Some(1.0)).unwrap();
        let clamped_data = clamped.data().as_f32_slice().unwrap();
        assert_eq!(clamped_data, &[-1.0, -0.5, 0.5, 1.0]);
        assert_eq!(clamped.shape(), &shape);
    }

    #[test]
    fn test_astype() {
        let data = vec![1.5f32, -2.3];
        let shape = Shape::new(vec![2]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let casted = tensor.astype(DataType::Float64).unwrap();
        let casted_data = casted.data().as_f64_slice().unwrap();
        assert!((casted_data[0] - 1.5).abs() < 1e-6);
        assert!((casted_data[1] - (-2.3)).abs() < 1e-6);
        assert_eq!(casted.shape(), &shape);

        let casted_int = tensor.astype(DataType::Int32).unwrap();
        let casted_int_data = casted_int.data().as_i32_slice().unwrap();
        assert_eq!(casted_int_data, &[1, -2]);
        assert_eq!(casted_int.shape(), &shape);

        let casted_bool = tensor.astype(DataType::Bool).unwrap();
        let casted_bool_data = casted_bool.data().as_bool_slice().unwrap();
        assert_eq!(casted_bool_data, &[true, true]);
    }

    #[test]
    fn test_astype_from_bool() {
        let data = vec![true, false, true];
        let shape = Shape::new(vec![3]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_bool(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let to_float = tensor.astype(DataType::Float32).unwrap();
        assert_eq!(to_float.data().as_f32_slice().unwrap(), &[1.0, 0.0, 1.0]);

        let to_int = tensor.astype(DataType::Int64).unwrap();
        assert_eq!(to_int.data().as_i64_slice().unwrap(), &[1, 0, 1]);
    }

    #[test]
    fn test_add_scalar_broadcasting() {
        let a = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let scalar = Tensor::ones(Shape::scalar(), DataType::Float32, Device::cpu(), false);
        let result = a.add(&scalar).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[2.0; 6]);
        assert_eq!(result.shape(), &Shape::new(vec![2, 3]));
    }

    #[test]
    fn test_add_incompatible_shapes_error() {
        let a = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let b = Tensor::ones(
            Shape::new(vec![3, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_view_shape_mismatch_error() {
        let shape = Shape::new(vec![2, 2]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape, DataType::Float32, Device::cpu(), false);
        let bad_shape = Shape::new(vec![3, 1]);
        assert!(tensor.view(bad_shape).is_err());
    }

    #[test]
    fn test_reshape_scalar_to_vector() {
        let scalar = Tensor::ones(Shape::scalar(), DataType::Float32, Device::cpu(), false);
        let reshaped = scalar.reshape(Shape::new(vec![1])).unwrap();
        assert_eq!(reshaped.shape().dims(), &[1]);
        assert_eq!(reshaped.data().as_f32_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn test_transpose_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(
            transposed.data().as_f32_slice().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_transpose_out_of_bounds() {
        let tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(tensor.transpose(0, 2).is_err());
    }

    #[test]
    fn test_transpose_same_dim_noop() {
        let tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let transposed = tensor.transpose(1, 1).unwrap();
        assert_eq!(transposed.data().as_f32_slice().unwrap(), &[1.0; 4]);
        assert_eq!(transposed.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_astype_multiple_conversions() {
        let base = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.5, -2.0, 0.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let as_i32 = base.astype(DataType::Int32).unwrap();
        assert_eq!(as_i32.data().as_i32_slice().unwrap(), &[1, -2, 0]);

        let as_bool = as_i32.astype(DataType::Bool).unwrap();
        assert_eq!(
            as_bool.data().as_bool_slice().unwrap(),
            &[true, true, false]
        );

        let as_f64 = as_bool.astype(DataType::Float64).unwrap();
        assert_eq!(as_f64.data().as_f64_slice().unwrap(), &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_astype_parallel_large_buffer() {
        let size = 2048;
        let data: Vec<f32> = (0..size).map(|v| v as f32).collect();
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![size]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let converted = tensor.astype(DataType::Int32).unwrap();
        let expected: Vec<i32> = (0..size).map(|v| v as i32).collect();
        assert_eq!(
            converted.data().as_i32_slice().unwrap(),
            expected.as_slice()
        );
    }

    #[test]
    fn test_array_equal_fast_path() {
        let t1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let t2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(t1.array_equal(&t2));
        assert!(t1.allclose(&t2, 0.0, 0.0));
    }

    #[test]
    fn test_array_equal_mismatch() {
        let t1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let t2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.1], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(!t1.array_equal(&t2));
        assert!(!t1.allclose(&t2, 1e-5, 1e-5));
    }

    #[test]
    fn test_array_equal_zero_sized() {
        let empty1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![], Device::cpu())),
            Shape::new(vec![0]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let empty2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![], Device::cpu())),
            Shape::new(vec![0]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(empty1.array_equal(&empty2));
    }

    #[test]
    fn test_deep_clone_independent_storage() {
        let shape = Shape::new(vec![2, 2]);
        let data = Arc::new(TensorData::from_vec_f32(
            vec![1.0, 2.0, 3.0, 4.0],
            Device::cpu(),
        ));
        let tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), false);

        let mut cloned = tensor.deep_clone().unwrap();
        {
            let slice = cloned.data_mut().as_f32_slice_mut().unwrap();
            slice[0] = 42.0;
        }

        let original_slice = tensor.data().as_f32_slice().unwrap();
        assert_eq!(original_slice, &[1.0, 2.0, 3.0, 4.0]);
        let cloned_slice = cloned.data().as_f32_slice().unwrap();
        assert_eq!(cloned_slice, &[42.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_deep_clone_preserves_gradients() {
        let shape = Shape::new(vec![3]);
        let data = Arc::new(TensorData::from_vec_f32(
            vec![1.0, -2.0, 3.0],
            Device::cpu(),
        ));
        let mut tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), true);
        tensor.zero_grad(true);

        let cloned = tensor.deep_clone().unwrap();
        assert!(cloned.requires_grad());

        let grad = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![0.5, -1.0, 2.0],
                Device::cpu(),
            )),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        );
        cloned.backward(Some(grad.clone())).unwrap();

        let accumulated = autograd::get_gradient(&tensor).expect("gradient should be set");
        assert!(accumulated.allclose(&grad, 1e-6, 1e-6));
    }

    #[test]
    fn test_contiguous_materialises_expanded_views() {
        let base = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0], Device::cpu())),
            Shape::new(vec![2, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let expanded = base
            .expand(vec![2isize, 3isize])
            .expect("expand should succeed");
        assert!(!expanded.is_contiguous());

        let contiguous = expanded.contiguous().expect("contiguous should copy data");
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims(), &[2, 3]);
        let values = contiguous
            .data()
            .as_f32_slice()
            .expect("materialised data should be accessible")
            .to_vec();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
}
