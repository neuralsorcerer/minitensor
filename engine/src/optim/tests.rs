// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(test)]
mod tests {
    use crate::optim::{
        Adam, GradientClipping, GradientUtils, Optimizer, ParameterGroup, RMSprop, SGD,
    };
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor},
    };

    #[test]
    fn test_sgd_creation() {
        let sgd = SGD::new(0.01, Some(0.9), Some(1e-4));
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.9);
        assert_eq!(sgd.weight_decay(), 1e-4);
        assert!(!sgd.is_nesterov());
    }

    #[test]
    fn test_sgd_with_options() {
        let sgd = SGD::new(0.01, Some(0.9), Some(1e-4))
            .with_nesterov(true)
            .with_dampening(0.1);

        assert!(sgd.is_nesterov());
        assert_eq!(sgd.momentum(), 0.9);
    }

    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8), Some(1e-4));
        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.beta1(), 0.9);
        assert_eq!(adam.beta2(), 0.999);
        assert_eq!(adam.epsilon(), 1e-8);
        assert!(!adam.is_amsgrad());
    }

    #[test]
    fn test_adam_with_amsgrad() {
        let adam = Adam::new(0.001, None, None, None, None).with_amsgrad(true);

        assert!(adam.is_amsgrad());
        assert_eq!(adam.beta1(), 0.9); // Default value
        assert_eq!(adam.beta2(), 0.999); // Default value
    }

    #[test]
    fn test_rmsprop_creation() {
        let rmsprop = RMSprop::new(0.01, Some(0.99), Some(1e-8), Some(1e-4), Some(0.9));
        assert_eq!(rmsprop.learning_rate(), 0.01);
        assert_eq!(rmsprop.alpha(), 0.99);
        assert_eq!(rmsprop.epsilon(), 1e-8);
        assert_eq!(rmsprop.momentum(), 0.9);
        assert!(!rmsprop.is_centered());
    }

    #[test]
    fn test_rmsprop_with_centered() {
        let rmsprop = RMSprop::new(0.01, None, None, None, None).with_centered(true);

        assert!(rmsprop.is_centered());
        assert_eq!(rmsprop.alpha(), 0.99); // Default value
    }

    #[test]
    fn test_optimizer_zero_grad() {
        let sgd = SGD::new(0.01, None, None);
        let mut tensor1 = Tensor::zeros(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let mut tensor2 = Tensor::zeros(
            Shape::new(vec![3, 3]),
            DataType::Float32,
            Device::cpu(),
            true,
        );

        // Set some gradients
        let grad1 = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grad2 = Tensor::ones(
            Shape::new(vec![3, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        tensor1.set_grad(Some(grad1));
        tensor2.set_grad(Some(grad2));

        assert!(tensor1.has_grad());
        assert!(tensor2.has_grad());

        let mut params = vec![&mut tensor1, &mut tensor2];
        sgd.zero_grad(&mut params, false).unwrap();

        assert!(tensor1.has_grad());
        assert!(tensor2.has_grad());
        let expected1 = Tensor::zeros(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let expected2 = Tensor::zeros(
            Shape::new(vec![3, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(tensor1.grad().unwrap().allclose(&expected1, 1e-6, 1e-6));
        assert!(tensor2.grad().unwrap().allclose(&expected2, 1e-6, 1e-6));
    }

    #[test]
    fn test_optimizer_zero_grad_set_to_none() {
        let sgd = SGD::new(0.01, None, None);
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let g = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(g.clone()));
        t2.set_grad(Some(g));
        let mut params = vec![&mut t1, &mut t2];
        sgd.zero_grad(&mut params, true).unwrap();
        assert!(t1.grad().is_none());
        assert!(t2.grad().is_none());
    }

    #[test]
    fn test_sgd_step_updates_parameters() {
        let mut sgd = SGD::new(0.1, None, None);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.step(&mut params).unwrap();
        let data = tensor.data().as_f32_slice().unwrap();
        // 1 - 0.1*1 = 0.9
        assert!(data.iter().all(|&v| (v - 0.9).abs() < 1e-6));
    }

    #[test]
    fn test_adam_step_updates_parameters() {
        let mut adam = Adam::new(0.1, Some(0.9), Some(0.999), Some(1e-8), None);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let mut grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 0.1);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        adam.step(&mut params).unwrap();
        let data = tensor.data().as_f32_slice().unwrap();
        // Expected ~0.9 as per Adam first update
        assert!(data.iter().all(|&v| (v - 0.9).abs() < 1e-5));
    }

    #[test]
    fn test_rmsprop_step_updates_parameters() {
        let mut rmsprop = RMSprop::new(0.1, Some(0.99), Some(1e-8), None, None);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let mut grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 0.1);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        rmsprop.step(&mut params).unwrap();
        let data = tensor.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| v < 1e-4));
    }

    #[test]
    fn test_sgd_nesterov_momentum_step() {
        let mut sgd = SGD::new(0.1, Some(0.9), None).with_nesterov(true);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.step(&mut params).unwrap();
        let data = tensor.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| (v - 0.81).abs() < 1e-6));
    }

    #[test]
    fn test_adam_amsgrad_effect() {
        let mut adam_plain = Adam::new(0.1, Some(0.9), Some(0.999), Some(1e-8), None);
        let mut adam_ams =
            Adam::new(0.1, Some(0.9), Some(0.999), Some(1e-8), None).with_amsgrad(true);
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        // First step with grad 0.1
        let mut g = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        g.data_mut().as_f32_slice_mut().unwrap()[0] = 0.1;
        t1.set_grad(Some(g.clone()));
        t2.set_grad(Some(g));
        let mut params1 = vec![&mut t1];
        let mut params2 = vec![&mut t2];
        adam_plain.step(&mut params1).unwrap();
        adam_ams.step(&mut params2).unwrap();
        // Second step with zero grad
        let g_zero = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(g_zero.clone()));
        t2.set_grad(Some(g_zero));
        let mut params1 = vec![&mut t1];
        let mut params2 = vec![&mut t2];
        adam_plain.step(&mut params1).unwrap();
        adam_ams.step(&mut params2).unwrap();
        let p_plain = t1.data().as_f32_slice().unwrap()[0];
        let p_ams = t2.data().as_f32_slice().unwrap()[0];
        assert!(p_ams > p_plain);
    }

    #[test]
    fn test_rmsprop_centered_momentum_step() {
        let mut rmsprop =
            RMSprop::new(0.1, Some(0.99), Some(1e-8), None, Some(0.9)).with_centered(true);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        tensor.set_grad(Some(grad));

        let mut params = vec![&mut tensor];
        rmsprop.step(&mut params).unwrap();
        let data = tensor.data().as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| v < 1.0));
    }

    #[test]
    fn test_gradient_clipping_by_norm() {
        let sgd = SGD::new(0.1, None, None)
            .with_gradient_clipping(GradientClipping::ByNorm { max_norm: 1.0 });
        let mut tensor = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 3.0);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        // Apply clipping directly
        sgd.clip_gradients(&mut params, &GradientClipping::ByNorm { max_norm: 1.0 })
            .unwrap();
        let norm = GradientUtils::compute_grad_norm(&[&tensor]).unwrap();
        assert!(norm <= 1.0001);
    }

    #[test]
    fn test_gradient_clipping_by_value() {
        let sgd = SGD::new(0.1, None, None).with_gradient_clipping(GradientClipping::ByValue {
            min_value: -1.0,
            max_value: 1.0,
        });
        let mut tensor = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::zeros(Shape::new(vec![3]), DataType::Float32, Device::cpu(), false);
        let gslice = grad.data_mut().as_f32_slice_mut().unwrap();
        gslice[0] = -2.0;
        gslice[1] = 0.5;
        gslice[2] = 2.0;
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.clip_gradients(
            &mut params,
            &GradientClipping::ByValue {
                min_value: -1.0,
                max_value: 1.0,
            },
        )
        .unwrap();
        let g = tensor.grad().unwrap().data().as_f32_slice().unwrap();
        assert!((g[0] + 1.0).abs() < 1e-6);
        assert!((g[1] - 0.5).abs() < 1e-6);
        assert!((g[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_float64() {
        let mut sgd = SGD::new(0.1, None, None);
        let mut tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float64,
            Device::cpu(),
            true,
        );
        let grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.step(&mut params).unwrap();
        let data = tensor.data().as_f64_slice().unwrap();
        assert!(data.iter().all(|&v| (v - 0.9).abs() < 1e-6));
    }

    #[test]
    fn test_parameter_groups() {
        use crate::autograd::TensorId;

        let param_group1 = ParameterGroup::new(vec![TensorId::new(), TensorId::new()], 0.01)
            .with_weight_decay(1e-4);
        let param_group2 = ParameterGroup::new(vec![TensorId::new()], 0.001);

        let mut sgd = SGD::with_param_groups(vec![param_group1, param_group2], 0.9);

        assert_eq!(sgd.param_groups().len(), 2);
        assert_eq!(sgd.param_groups()[0].lr, 0.01);
        assert_eq!(sgd.param_groups()[1].lr, 0.001);
        assert_eq!(sgd.param_groups()[0].weight_decay, 1e-4);
        assert_eq!(sgd.param_groups()[1].weight_decay, 0.0);

        // Test adding a new parameter group
        let param_group3 = ParameterGroup::new(vec![TensorId::new()], 0.1);
        sgd.add_param_group(param_group3).unwrap();
        assert_eq!(sgd.param_groups().len(), 3);
    }

    #[test]
    fn test_learning_rate_modification() {
        let mut sgd = SGD::new(0.01, None, None);
        assert_eq!(sgd.learning_rate(), 0.01);

        sgd.set_learning_rate(0.001);
        assert_eq!(sgd.learning_rate(), 0.001);

        let mut adam = Adam::new(0.001, None, None, None, None);
        assert_eq!(adam.learning_rate(), 0.001);

        adam.set_learning_rate(0.0001);
        assert_eq!(adam.learning_rate(), 0.0001);
    }

    #[test]
    fn test_step_count() {
        let sgd = SGD::new(0.01, None, None);
        assert_eq!(sgd.step_count(), 0);

        let adam = Adam::new(0.001, None, None, None, None);
        assert_eq!(adam.step_count(), 0);

        let rmsprop = RMSprop::new(0.01, None, None, None, None);
        assert_eq!(rmsprop.step_count(), 0);
    }

    #[test]
    fn test_gradient_clipping_by_norm_no_change() {
        let sgd = SGD::new(0.1, None, None);
        let mut tensor = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.clip_gradients(&mut params, &GradientClipping::ByNorm { max_norm: 10.0 })
            .unwrap();
        let g = tensor.grad().unwrap().data().as_f32_slice().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-6 && (g[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_weight_decay() {
        let mut sgd = SGD::new(0.1, None, Some(0.5));
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.step(&mut params).unwrap();
        let val = tensor.data().as_f32_slice().unwrap()[0];
        assert!((val - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_adam_weight_decay() {
        let mut adam = Adam::new(0.1, None, None, None, Some(0.5));
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        adam.step(&mut params).unwrap();
        let val = tensor.data().as_f32_slice().unwrap()[0];
        assert!((val - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_step_without_gradient() {
        let mut sgd = SGD::new(0.1, None, None);
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut params = vec![&mut tensor];
        sgd.step(&mut params).unwrap();
        assert_eq!(sgd.step_count(), 1);
        let val = tensor.data().as_f32_slice().unwrap()[0];
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_param_group_updates() {
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(grad.clone()));
        t2.set_grad(Some(grad));
        let id1 = t1.id();
        let id2 = t2.id();
        let g1 = ParameterGroup::new(vec![id1], 0.5);
        let g2 = ParameterGroup::new(vec![id2], 0.1);
        let mut sgd = SGD::with_param_groups(vec![g1, g2], 0.0);
        let mut params = vec![&mut t1, &mut t2];
        sgd.step(&mut params).unwrap();
        let v1 = t1.data().as_f32_slice().unwrap()[0];
        let v2 = t2.data().as_f32_slice().unwrap()[0];
        assert!((v1 - 0.5).abs() < 1e-6);
        assert!((v2 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_adam_param_group_weight_decay() {
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let zero_grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(zero_grad.clone()));
        t2.set_grad(Some(zero_grad));
        let id1 = t1.id();
        let id2 = t2.id();
        let g1 = ParameterGroup::new(vec![id1], 0.1).with_weight_decay(0.1);
        let g2 = ParameterGroup::new(vec![id2], 0.1);
        let mut adam = Adam::with_param_groups(vec![g1, g2], 0.9, 0.999, 1e-8);
        let mut params = vec![&mut t1, &mut t2];
        adam.step(&mut params).unwrap();
        let v1 = t1.data().as_f32_slice().unwrap()[0];
        let v2 = t2.data().as_f32_slice().unwrap()[0];
        assert!(v1 < v2);
    }

    #[test]
    fn test_rmsprop_param_group_learning_rates() {
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(grad.clone()));
        t2.set_grad(Some(grad));
        let id1 = t1.id();
        let id2 = t2.id();
        let g1 = ParameterGroup::new(vec![id1], 0.5);
        let g2 = ParameterGroup::new(vec![id2], 0.1);
        let mut rms = RMSprop::with_param_groups(vec![g1, g2], 0.99, 1e-8, 0.0);
        let mut params = vec![&mut t1, &mut t2];
        rms.step(&mut params).unwrap();
        let v1 = t1.data().as_f32_slice().unwrap()[0];
        let v2 = t2.data().as_f32_slice().unwrap()[0];
        assert!(v1 < v2);
    }

    #[test]
    fn test_zero_learning_rate_no_update() {
        let mut sgd = SGD::new(0.0, None, None);
        let mut adam = Adam::new(0.0, None, None, None, None);
        let mut rms = RMSprop::new(0.0, None, None, None, None);
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t3 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(grad.clone()));
        t2.set_grad(Some(grad.clone()));
        t3.set_grad(Some(grad));
        sgd.step(&mut [&mut t1]).unwrap();
        adam.step(&mut [&mut t2]).unwrap();
        rms.step(&mut [&mut t3]).unwrap();
        assert!((t1.data().as_f32_slice().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((t2.data().as_f32_slice().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((t3.data().as_f32_slice().unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping_by_norm_float64() {
        let sgd = SGD::new(0.1, None, None)
            .with_gradient_clipping(GradientClipping::ByNorm { max_norm: 1.0 });
        let mut tensor = Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        grad.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 3.0);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.clip_gradients(&mut params, &GradientClipping::ByNorm { max_norm: 1.0 })
            .unwrap();
        let norm = GradientUtils::compute_grad_norm(&[&tensor]).unwrap();
        assert!(norm <= 1.0001);
    }

    #[test]
    fn test_gradient_clipping_by_value_float64() {
        let sgd = SGD::new(0.1, None, None).with_gradient_clipping(GradientClipping::ByValue {
            min_value: -0.5,
            max_value: 0.5,
        });
        let mut tensor = Tensor::ones(Shape::new(vec![3]), DataType::Float64, Device::cpu(), true);
        let mut grad = Tensor::zeros(Shape::new(vec![3]), DataType::Float64, Device::cpu(), false);
        let g = grad.data_mut().as_f64_slice_mut().unwrap();
        g[0] = -1.0;
        g[1] = 0.2;
        g[2] = 1.0;
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        sgd.clip_gradients(
            &mut params,
            &GradientClipping::ByValue {
                min_value: -0.5,
                max_value: 0.5,
            },
        )
        .unwrap();
        let g = tensor.grad().unwrap().data().as_f64_slice().unwrap();
        assert!((g[0] + 0.5).abs() < 1e-12);
        assert!((g[1] - 0.2).abs() < 1e-12);
        assert!((g[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_adam_add_param_group_updates_learning_rate() {
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let id = tensor.id();
        let mut adam = Adam::new(0.1, None, None, None, None);
        let group = ParameterGroup::new(vec![id], 0.01);
        adam.add_param_group(group).unwrap();
        adam.step(&mut [&mut tensor]).unwrap();
        let val = tensor.data().as_f32_slice().unwrap()[0];
        assert!((val - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_add_param_group_updates_learning_rate() {
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let id = tensor.id();
        let mut rms = RMSprop::new(0.1, None, None, None, None);
        let group = ParameterGroup::new(vec![id], 0.01);
        rms.add_param_group(group).unwrap();
        rms.step(&mut [&mut tensor]).unwrap();
        let val = tensor.data().as_f32_slice().unwrap()[0];
        assert!((val - 0.9).abs() < 1e-6);
    }
}
