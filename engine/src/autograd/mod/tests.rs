// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

/// Helper function to reduce gradients for broadcasting
fn reduce_gradient_for_broadcasting(grad_output: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    if grad_output.shape() == target_shape {
        return Ok(grad_output.clone());
    }

    let grad_dims = grad_output.shape().dims();
    let target_dims = target_shape.dims();
    if target_dims.len() > grad_dims.len() {
        return Err(MinitensorError::BroadcastError {
            shape1: grad_dims.to_vec(),
            shape2: target_dims.to_vec(),
            suggestion: Some(
                "Ensure the target shape has no more dimensions than the gradient output."
                    .to_string(),
            ),
            context: Some("reduce_gradient_for_broadcasting".to_string()),
        });
    }
    let extra = grad_dims.len() - target_dims.len();

    // Use a stack-allocated small vector and pre-allocate enough capacity to
    // hold all potential broadcast axes. This avoids repeated reallocations for
    // higher dimensional tensors.
    let mut axes_to_sum: SmallVec<[usize; 8]> = SmallVec::with_capacity(grad_dims.len());
    axes_to_sum.extend(0..extra);
    for i in 0..target_dims.len() {
        let gdim = grad_dims[extra + i];
        let tdim = target_dims[i];
        if tdim == 1 {
            if gdim != 1 {
                axes_to_sum.push(extra + i);
            }
        } else if gdim != tdim {
            return Err(MinitensorError::BroadcastError {
                shape1: grad_dims.to_vec(),
                shape2: target_dims.to_vec(),
                suggestion: Some(
                    "Ensure each target dimension is 1 or matches the gradient dimension."
                        .to_string(),
                ),
                context: Some("reduce_gradient_for_broadcasting".to_string()),
            });
        }
    }

    if axes_to_sum.is_empty() {
        return Ok(grad_output.clone());
    }

    let mut axes = Vec::with_capacity(axes_to_sum.len());
    for axis in axes_to_sum {
        axes.push(axis as isize);
    }
    let mut grad = reduction::sum(grad_output, Some(axes), true)?;

    if grad.shape() != target_shape {
        grad = grad.view(target_shape.clone())?;
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::DataType;

    #[test]
    fn test_tensor_id_generation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_computation_graph() {
        let mut graph = ComputationGraph::new();
        let tensor_id = TensorId::new();

        let grad_fn = Arc::new(AddBackward {
            input_shapes: [vec![2, 2], vec![2, 2]],
            input_ids: [TensorId::new(), TensorId::new()],
        });

        graph.add_tensor_with_grad_req(tensor_id, Some(grad_fn), true);
        assert!(graph.nodes().contains_key(&tensor_id));
    }

    #[test]
    fn test_add_backward() {
        let grad_fn = AddBackward {
            input_shapes: [vec![2, 2], vec![2, 2]],
            input_ids: [TensorId::new(), TensorId::new()],
        };

        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            crate::tensor::DataType::Float32,
            Device::cpu(),
            false,
        );
        let gradients = grad_fn.backward(&grad_output).unwrap();

        assert_eq!(gradients.len(), 2);
    }

    #[test]
    fn test_reduce_gradient_for_broadcasting() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![2, 1]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 1]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!(slice.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_reduce_gradient_multiple_axes() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![1, 1]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[1, 1]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!((slice[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_gradient_with_leading_and_inner_axes() {
        let grad_output = Tensor::ones(
            Shape::new(vec![4, 2, 3, 5]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![2, 1, 5]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 1, 5]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!(slice.iter().all(|&x| (x - 12.0).abs() < 1e-6));
    }

    #[test]
    fn test_reduce_gradient_noop_for_same_shape() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = grad_output.shape().clone();
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 3, 4]);
        assert!(reduced.allclose(&grad_output, 1e-6, 1e-6));
    }

    #[test]
    fn test_reduce_gradient_invalid_broadcast() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![2, 2]);
        let err = reduce_gradient_for_broadcasting(&grad_output, &target_shape)
            .expect_err("expected invalid broadcast error");
        assert!(matches!(err, MinitensorError::BroadcastError { .. }));
    }

    #[test]
    fn test_reduce_gradient_zero_dim_broadcast() {
        let grad_output = Tensor::ones(
            Shape::new(vec![0, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![1, 2]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[1, 2]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!(slice.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_softmax_backward_dim1() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let softmax_out = activation::softmax(&input, Some(1)).unwrap();
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grad_y = arithmetic::mul(&grad_output, &softmax_out).unwrap();
        let sum = reduction::sum(&grad_y, Some(vec![1]), true).unwrap();
        let sub = arithmetic::sub(&grad_output, &sum).unwrap();
        let expected = arithmetic::mul(&softmax_out, &sub).unwrap();

        let grad_fn = SoftmaxBackward {
            input_id: TensorId::new(),
            output: softmax_out.clone(),
            dim: 1,
        };
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_softmax_backward_dim0_f64() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let softmax_out = activation::softmax(&input, Some(0)).unwrap();
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let grad_y = arithmetic::mul(&grad_output, &softmax_out).unwrap();
        let sum = reduction::sum(&grad_y, Some(vec![0]), true).unwrap();
        let sub = arithmetic::sub(&grad_output, &sum).unwrap();
        let expected = arithmetic::mul(&softmax_out, &sub).unwrap();

        let grad_fn = SoftmaxBackward {
            input_id: TensorId::new(),
            output: softmax_out.clone(),
            dim: 0,
        };
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_backward_broadcast_addition() {
        clear_graph().unwrap();

        let a = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let b = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let out = arithmetic::add(&a, &b).unwrap();

        let grad = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward(&out, Some(grad)).unwrap();

        let grad_a = grads.get(&a.id()).unwrap();
        let grad_b = grads.get(&b.id()).unwrap();
        assert_eq!(grad_a.shape().dims(), &[2, 3]);
        assert_eq!(grad_b.shape().dims(), &[3]);
        let slice_b = grad_b.data().as_f32_slice().unwrap();
        assert!(slice_b.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_matmul_backward_gradients() {
        let lhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let input_ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids,
            lhs_requires_grad: true,
            rhs_requires_grad: true,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let rhs_t = crate::operations::linalg::transpose(&rhs, 0, 1).unwrap();
        let expected_lhs = crate::operations::linalg::matmul(&grad_output, &rhs_t).unwrap();
        let lhs_grad = grads.get(&input_ids[0]).unwrap();
        assert!(lhs_grad.allclose(&expected_lhs, 1e-6, 1e-6));
    }

    #[test]
    fn test_matmul_backward_batched() {
        let lhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..12).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..24).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let input_ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids,
            lhs_requires_grad: true,
            rhs_requires_grad: true,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let rhs_t = crate::operations::linalg::transpose(
            &rhs,
            (rhs.ndim() - 2) as isize,
            (rhs.ndim() - 1) as isize,
        )
        .unwrap();
        let expected_lhs = crate::operations::linalg::matmul(&grad_output, &rhs_t).unwrap();
        assert!(
            grads
                .get(&input_ids[0])
                .unwrap()
                .allclose(&expected_lhs, 1e-6, 1e-6)
        );
        let lhs_t = crate::operations::linalg::transpose(
            &lhs,
            (lhs.ndim() - 2) as isize,
            (lhs.ndim() - 1) as isize,
        )
        .unwrap();
        let expected_rhs = crate::operations::linalg::matmul(&lhs_t, &grad_output).unwrap();
        assert!(
            grads
                .get(&input_ids[1])
                .unwrap()
                .allclose(&expected_rhs, 1e-6, 1e-6)
        );
    }

    #[test]
    fn test_matmul_backward_requires_grad_flags() {
        let lhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: ids,
            lhs_requires_grad: true,
            rhs_requires_grad: false,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        assert!(grads.contains_key(&ids[0]));
        assert!(!grads.contains_key(&ids[1]));
    }

    #[test]
    fn test_transpose_backward_permutation() {
        let _input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..24).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let dims = vec![1, 2, 0];
        let grad_fn = TransposeBackward {
            dims: dims.clone(),
            input_id: TensorId::new(),
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![3, 4, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        let mut inverse = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inverse[d] = i;
        }
        let mut expected = grad_output.clone();
        let mut current: Vec<usize> = (0..inverse.len()).collect();
        for i in 0..inverse.len() {
            let j = current.iter().position(|&x| x == inverse[i]).unwrap();
            if i != j {
                expected = crate::operations::linalg::transpose(&expected, i as isize, j as isize)
                    .unwrap();
                current.swap(i, j);
            }
        }
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }
}
