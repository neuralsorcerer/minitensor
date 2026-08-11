// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(test)]
#[allow(clippy::module_inception)] // file is already the `tests` module of `optim`
mod tests {
    use crate::optim::optimizer::LearningRateScheduler;
    use crate::optim::{
        Adagrad, Adam, AdamW, CosineAnnealingLR, GradientClipping, GradientUtils, NAdam, Optimizer,
        ParameterGroup, RMSprop, SGD,
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
    fn test_sgd_momentum_dampening_first_step_matches_pytorch() {
        // The momentum buffer is seeded with the raw gradient on the first
        // step (`buf = grad.clone()`), applying the (1 - dampening) factor only
        // from the second step onward. Verify both steps against hand-computed
        // reference values (lr=0.1, momentum=0.9, dampening=0.5, grad=2.0).
        let mut p = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        grad.data_mut().as_f32_slice_mut().unwrap()[0] = 2.0;

        let mut sgd = SGD::new(0.1, Some(0.9), Some(0.0)).with_dampening(0.5);

        // Step 1: buf = grad = 2.0  ->  p = 1.0 - 0.1 * 2.0 = 0.8
        p.set_grad(Some(grad.clone()));
        {
            let mut params = vec![&mut p];
            sgd.step(&mut params).unwrap();
        }
        let after1 = p.data().as_f32_slice().unwrap()[0];
        assert!((after1 - 0.8).abs() < 1e-6, "first step got {after1}");

        // Step 2: buf = 0.9 * 2.0 + 0.5 * 2.0 = 2.8  ->  p = 0.8 - 0.1 * 2.8 = 0.52
        p.set_grad(Some(grad.clone()));
        {
            let mut params = vec![&mut p];
            sgd.step(&mut params).unwrap();
        }
        let after2 = p.data().as_f32_slice().unwrap()[0];
        assert!((after2 - 0.52).abs() < 1e-6, "second step got {after2}");
    }

    #[test]
    fn test_rmsprop_momentum_lr_schedule_matches_pytorch() {
        // lr is kept out of the RMSprop momentum buffer:
        //   buf = momentum*buf + grad/denom ; param -= lr*buf
        // so a mid-training lr change rescales the entire accumulated buffer.
        // Reference (f64, alpha=0.99, eps=1e-8, momentum=0.9, grad=2.0):
        //   step1 lr=0.1: sq=0.04, denom≈0.2, buf≈10, p≈1-0.1*10=0.0
        //   step2 lr=0.5: sq≈0.0796, denom≈0.28213, buf≈0.9*10+2/0.28213≈16.0888,
        //                 p≈0.0-0.5*16.0888≈-8.0444
        let mut p = Tensor::ones(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);
        let mut grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), false);
        grad.data_mut().as_f64_slice_mut().unwrap()[0] = 2.0;

        let mut opt = RMSprop::new(0.1, Some(0.99), Some(1e-8), Some(0.0), Some(0.9));

        p.set_grad(Some(grad.clone()));
        {
            let mut params = vec![&mut p];
            opt.step(&mut params).unwrap();
        }
        let after1 = p.data().as_f64_slice().unwrap()[0];
        assert!(after1.abs() < 1e-3, "first step got {after1}");

        opt.set_learning_rate(0.5);
        p.set_grad(Some(grad.clone()));
        {
            let mut params = vec![&mut p];
            opt.step(&mut params).unwrap();
        }
        let after2 = p.data().as_f64_slice().unwrap()[0];
        assert!((after2 + 8.0444).abs() < 1e-2, "second step got {after2}");
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
    fn test_adamw_creation() {
        let adamw = AdamW::new(0.001, Some(0.9), Some(0.999), Some(1e-8), Some(0.01));
        assert_eq!(adamw.learning_rate(), 0.001);
        assert_eq!(adamw.beta1(), 0.9);
        assert_eq!(adamw.beta2(), 0.999);
        assert_eq!(adamw.epsilon(), 1e-8);
        assert_eq!(adamw.weight_decay(), 0.01);
    }

    #[test]
    fn test_adamw_param_group_weight_decay() {
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);

        // Zero gradients so only weight decay contributes to the update.
        let zero_grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        t1.set_grad(Some(zero_grad.clone()));
        t2.set_grad(Some(zero_grad));

        let id1 = t1.id();
        let id2 = t2.id();

        // Distinct weight decays per parameter group to verify decoupled handling.
        let g1 = ParameterGroup::new(vec![id1], 0.1).with_weight_decay(0.4);
        let g2 = ParameterGroup::new(vec![id2], 0.1).with_weight_decay(0.0);

        let mut adamw = AdamW::with_param_groups(vec![g1, g2], 0.9, 0.999, 1e-8);

        let mut params = vec![&mut t1, &mut t2];
        adamw.step(&mut params).unwrap();

        let v1 = t1.data().as_f32_slice().unwrap()[0];
        let v2 = t2.data().as_f32_slice().unwrap()[0];

        // With decoupled decay: p = p - lr * wd * p.
        let expected_v1 = 1.0 - 0.1 * 0.4;
        let expected_v2 = 1.0; // No weight decay on second group

        assert!((v1 - expected_v1).abs() < 1e-6);
        assert!((v2 - expected_v2).abs() < 1e-6);
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

    /// One RMSprop step from `p = 1`, `g = 2` with `alpha = 0.9`, `lr = 0.1`,
    /// no weight decay, for a tensor of `numel` elements.
    fn rmsprop_one_step(centered: bool, momentum: f64, numel: usize) -> Vec<f32> {
        let mut opt = RMSprop::new(0.1, Some(0.9), Some(1e-8), Some(0.0), Some(momentum))
            .with_centered(centered);
        let mut p = Tensor::ones(
            Shape::new(vec![numel]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let mut g = Tensor::ones(
            Shape::new(vec![numel]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        g.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 2.0);
        p.set_grad(Some(g));
        opt.step(&mut [&mut p]).unwrap();
        p.data().as_f32_slice().unwrap().to_vec()
    }

    #[test]
    fn test_rmsprop_all_state_combinations_match_reference() {
        // The four (momentum buffer × centering) combinations used to be four
        // separate kernels per dtype; they now share one. Pin each against a
        // hand-computed step so the shared path cannot drift.
        //   sq = 0.9*0 + 0.1*g^2 = 0.4
        //   plain/momentum:  denom = sqrt(0.4)
        //   centered:        ga = 0.2, var = 0.4 - 0.04 = 0.36, denom = 0.6
        let sqrt_04 = 0.4f32.sqrt();
        let cases: &[(bool, f64, f32)] = &[
            (false, 0.0, 1.0 - 0.1 * 2.0 / sqrt_04),
            (false, 0.9, 1.0 - 0.1 * (2.0 / sqrt_04)),
            (true, 0.0, 1.0 - 0.1 * 2.0 / 0.6),
            (true, 0.9, 1.0 - 0.1 * (2.0 / 0.6)),
        ];
        for &(centered, momentum, expected) in cases {
            let got = rmsprop_one_step(centered, momentum, 4)[0];
            assert!(
                (got - expected).abs() < 1e-5,
                "centered={centered} momentum={momentum}: got {got}, want {expected}"
            );
        }
    }

    #[test]
    fn test_rmsprop_parallel_path_matches_sequential() {
        // Above `PAR_THRESHOLD` the update runs over rayon chunks; the result
        // must be identical to the single-threaded path element for element.
        for &(centered, momentum) in &[(false, 0.0), (false, 0.9), (true, 0.0), (true, 0.9)] {
            let small = rmsprop_one_step(centered, momentum, 8)[0];
            let large = rmsprop_one_step(centered, momentum, 10_000);
            assert_eq!(large.len(), 10_000);
            for (i, &v) in large.iter().enumerate() {
                assert!(
                    (v - small).abs() < 1e-6,
                    "centered={centered} momentum={momentum} index {i}: {v} vs {small}"
                );
            }
        }
    }

    #[test]
    fn test_optimizers_parallel_path_matches_sequential() {
        // Same cross-check for the other optimizers' chunked updates.
        fn one_step<F>(numel: usize, make: F) -> Vec<f32>
        where
            F: FnOnce() -> Box<dyn Optimizer>,
        {
            let mut opt = make();
            let mut p = Tensor::ones(
                Shape::new(vec![numel]),
                DataType::Float32,
                Device::cpu(),
                true,
            );
            let mut g = Tensor::ones(
                Shape::new(vec![numel]),
                DataType::Float32,
                Device::cpu(),
                false,
            );
            g.data_mut()
                .as_f32_slice_mut()
                .unwrap()
                .iter_mut()
                .for_each(|v| *v = 2.0);
            p.set_grad(Some(g));
            opt.step(&mut [&mut p]).unwrap();
            p.data().as_f32_slice().unwrap().to_vec()
        }

        type Builder = fn() -> Box<dyn Optimizer>;
        let builders: Vec<(&str, Builder)> = vec![
            ("sgd", || Box::new(SGD::new(0.1, None, Some(0.01)))),
            ("sgd_momentum", || {
                Box::new(SGD::new(0.1, Some(0.9), Some(0.01)))
            }),
            ("adam", || {
                Box::new(Adam::new(0.1, None, None, None, Some(0.01)))
            }),
            ("adamw", || {
                Box::new(AdamW::new(0.1, None, None, None, Some(0.01)))
            }),
            ("adam_amsgrad", || {
                Box::new(Adam::new(0.1, None, None, None, None).with_amsgrad(true))
            }),
            ("lion", || {
                Box::new(crate::optim::Lion::new(0.1, None, None, Some(0.01)))
            }),
        ];

        for (name, build) in builders {
            let small = one_step(8, build)[0];
            let large = one_step(10_000, build);
            for (i, &v) in large.iter().enumerate() {
                assert!((v - small).abs() < 1e-6, "{name} index {i}: {v} vs {small}");
            }
        }
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
    fn test_adamw_decoupled_weight_decay() {
        let mut adamw = AdamW::new(0.1, None, None, None, Some(0.5));
        let mut tensor = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        let mut params = vec![&mut tensor];
        adamw.step(&mut params).unwrap();
        let val = tensor.data().as_f32_slice().unwrap()[0];
        // With decoupled weight decay: p = p - lr * wd * p = 1 - 0.1*0.5
        assert!((val - 0.95).abs() < 1e-6);
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

    #[test]
    fn test_cosine_annealing_scheduler_bounds() {
        let scheduler = CosineAnnealingLR::new(4, 0.01);
        let base_lr = 0.1;

        let lr_start = scheduler.get_lr(0, base_lr);
        assert!((lr_start - base_lr).abs() < 1e-12);

        let lr_mid = scheduler.get_lr(2, base_lr);
        let expected_mid = 0.01 + (base_lr - 0.01) * 0.5;
        assert!((lr_mid - expected_mid).abs() < 1e-12);

        let lr_end = scheduler.get_lr(4, base_lr);
        assert!((lr_end - 0.01).abs() < 1e-12);

        let lr_after = scheduler.get_lr(10, base_lr);
        assert!((lr_after - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_annealing_zero_t_max() {
        let scheduler = CosineAnnealingLR::new(0, 0.01);
        let base_lr = 0.1;
        assert!((scheduler.get_lr(0, base_lr) - base_lr).abs() < 1e-12);
        assert!((scheduler.get_lr(5, base_lr) - base_lr).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_utils_with_missing_gradients() {
        let mut with_grad =
            Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let without_grad =
            Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        with_grad.set_grad(Some(grad));

        let params = [&with_grad, &without_grad];
        assert!(GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 1);

        let norm = GradientUtils::compute_grad_norm(&params).unwrap();
        assert!((norm - (2.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_utils_when_all_gradients_absent() {
        let t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);
        let params = [&t1, &t2];

        assert!(!GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 0);
        assert_eq!(GradientUtils::compute_grad_norm(&params).unwrap(), 0.0);
    }

    #[test]
    fn test_clip_grad_value_no_gradient_is_noop() {
        let mut t = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        GradientUtils::clip_grad_value(&mut [&mut t], -1.0, 1.0).unwrap();
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_scheduler_utils_variants_and_edge_steps() {
        let warmup = crate::optim::SchedulerUtils::linear_warmup(4);
        assert!((warmup.get_lr(0, 0.2) - 0.0).abs() < 1e-12);
        assert!((warmup.get_lr(2, 0.2) - 0.1).abs() < 1e-12);
        assert!((warmup.get_lr(4, 0.2) - 0.2).abs() < 1e-12);

        let warmup_zero = crate::optim::SchedulerUtils::linear_warmup(0);
        assert!((warmup_zero.get_lr(0, 0.2) - 0.2).abs() < 1e-12);

        let poly = crate::optim::SchedulerUtils::polynomial_decay(5, 0.01, 2.0);
        let lr_step_0 = poly.get_lr(0, 0.11);
        let lr_step_3 = poly.get_lr(3, 0.11);
        let lr_after_decay = poly.get_lr(5, 0.11);
        assert!((lr_step_0 - 0.11).abs() < 1e-12);
        assert!(lr_step_3 < lr_step_0);
        assert!((lr_after_decay - 0.01).abs() < 1e-12);

        let multi = crate::optim::SchedulerUtils::multi_step(vec![6, 2, 4], 0.5);
        assert!((multi.get_lr(1, 0.2) - 0.2).abs() < 1e-12);
        assert!((multi.get_lr(2, 0.2) - 0.1).abs() < 1e-12);
        assert!((multi.get_lr(4, 0.2) - 0.05).abs() < 1e-12);
        assert!((multi.get_lr(6, 0.2) - 0.025).abs() < 1e-12);
    }

    #[test]
    fn test_compute_grad_norm_float64_values() {
        let mut t = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut g = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        let data = g.data_mut().as_f64_slice_mut().unwrap();
        data[0] = 3.0;
        data[1] = 4.0;
        t.set_grad(Some(g));

        let norm = GradientUtils::compute_grad_norm(&[&t]).unwrap();
        assert!((norm - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_grad_norm_ignores_non_float_gradient_dtype() {
        let mut t = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let mut int_grad = Tensor::ones(Shape::new(vec![3]), DataType::Int32, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 7);
        t.set_grad(Some(int_grad));

        let norm = GradientUtils::compute_grad_norm(&[&t]).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_norm_returns_original_norm_without_clipping() {
        let mut t = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut g = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        let grad = g.data_mut().as_f32_slice_mut().unwrap();
        grad[0] = 0.3;
        grad[1] = 0.4;
        t.set_grad(Some(g));

        let norm = GradientUtils::clip_grad_norm(&mut [&mut t], 1.0).unwrap();
        assert!((norm - 0.5).abs() < 1e-6);

        let grad_after = t.grad().unwrap().data().as_f32_slice().unwrap();
        assert!((grad_after[0] - 0.3).abs() < 1e-6);
        assert!((grad_after[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_scales_float64_gradient() {
        let mut t = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut g = Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        g.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 3.0);
        t.set_grad(Some(g));

        let norm_before = GradientUtils::compute_grad_norm(&[&t]).unwrap();
        assert!((norm_before - (18.0f64).sqrt()).abs() < 1e-12);

        let returned_norm = GradientUtils::clip_grad_norm(&mut [&mut t], 1.0).unwrap();
        assert!((returned_norm - norm_before).abs() < 1e-12);

        let norm_after = GradientUtils::compute_grad_norm(&[&t]).unwrap();
        assert!(norm_after <= 1.0001);
    }

    #[test]
    fn test_clip_grad_norm_ignores_non_float_gradient_dtype() {
        let mut t = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut int_grad = Tensor::ones(Shape::new(vec![2]), DataType::Int64, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i64_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = 10);
        t.set_grad(Some(int_grad));

        let norm = GradientUtils::clip_grad_norm(&mut [&mut t], 0.5).unwrap();
        assert_eq!(norm, 0.0);

        let grad_after = t.grad().unwrap().data().as_i64_slice().unwrap();
        assert_eq!(grad_after, &[10, 10]);
    }

    #[test]
    fn test_compute_grad_norm_accumulates_across_f32_and_f64_params() {
        let mut t1 = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut g1 = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        let g1s = g1.data_mut().as_f32_slice_mut().unwrap();
        g1s[0] = 1.0;
        g1s[1] = 2.0;
        t1.set_grad(Some(g1));

        let mut t2 = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut g2 = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        let g2s = g2.data_mut().as_f64_slice_mut().unwrap();
        g2s[0] = 2.0;
        g2s[1] = 1.0;
        t2.set_grad(Some(g2));

        let norm = GradientUtils::compute_grad_norm(&[&t1, &t2]).unwrap();
        assert!((norm - 10.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_clip_grad_norm_returns_zero_when_no_gradients_exist() {
        let mut t1 = Tensor::ones(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        let mut t2 = Tensor::ones(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);

        let returned = GradientUtils::clip_grad_norm(&mut [&mut t1, &mut t2], 0.1).unwrap();
        assert_eq!(returned, 0.0);
        assert!(t1.grad().is_none());
        assert!(t2.grad().is_none());
    }

    #[test]
    fn test_scheduler_utils_polynomial_decay_with_zero_decay_steps() {
        let scheduler = crate::optim::SchedulerUtils::polynomial_decay(0, 0.05, 2.0);
        assert!((scheduler.get_lr(0, 0.2) - 0.05).abs() < 1e-12);
        assert!((scheduler.get_lr(10, 0.2) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_scheduler_utils_multi_step_with_duplicate_milestones() {
        let scheduler = crate::optim::SchedulerUtils::multi_step(vec![2, 2, 4], 0.5);
        assert!((scheduler.get_lr(1, 0.2) - 0.2).abs() < 1e-12);
        assert!((scheduler.get_lr(2, 0.2) - 0.05).abs() < 1e-12);
        assert!((scheduler.get_lr(4, 0.2) - 0.025).abs() < 1e-12);
    }

    #[test]
    fn test_scheduler_utils_linear_warmup_single_step_transition() {
        let scheduler = crate::optim::SchedulerUtils::linear_warmup(1);
        assert!((scheduler.get_lr(0, 0.3) - 0.0).abs() < 1e-12);
        assert!((scheduler.get_lr(1, 0.3) - 0.3).abs() < 1e-12);
        assert!((scheduler.get_lr(2, 0.3) - 0.3).abs() < 1e-12);
    }

    /// Build `y = sum(x * x)` and run backward, so the gradient for `x` lives
    /// in the autograd graph (`autograd::get_gradient`) rather than in the
    /// tensor-local `.grad` — the arrangement every real training loop uses.
    fn param_with_graph_gradient(value: f32) -> Tensor {
        use crate::ops::{arithmetic::mul, reduction::sum};

        crate::autograd::clear_graph().unwrap();
        let mut x = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        x.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|v| *v = value);
        let y = sum(&mul(&x, &x).unwrap(), None, false).unwrap();
        crate::autograd::backward(&y, None).unwrap();
        assert!(x.grad().is_none(), "gradient should live in the graph only");
        x
    }

    #[test]
    fn test_clip_grad_norm_clips_graph_stored_gradients() {
        // Regression: clipping used to read only the tensor-local `.grad`,
        // so after a real `backward()` it silently did nothing while
        // `step()` went on to consume the unclipped graph gradient.
        let mut x = param_with_graph_gradient(3.0); // grad = 2*x = [6, 6, 6]
        let before = GradientUtils::compute_grad_norm(&[&x]).unwrap();
        assert!((before - (108.0f64).sqrt()).abs() < 1e-6, "got {before}");

        GradientUtils::clip_grad_norm(&mut [&mut x], 1.0).unwrap();

        let after = GradientUtils::compute_grad_norm(&[&x]).unwrap();
        assert!(after <= 1.0 + 1e-4, "norm not clipped: {after}");
        // The optimizer's own view of the gradient must be the clipped one.
        let stored = crate::autograd::get_gradient(&x).unwrap();
        let values = stored.data().as_f32_slice().unwrap();
        for &v in values {
            assert!((v - 1.0 / (3.0f32).sqrt()).abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn test_clip_grad_value_clips_graph_stored_gradients() {
        let mut x = param_with_graph_gradient(3.0); // grad = [6, 6, 6]
        GradientUtils::clip_grad_value(&mut [&mut x], -1.0, 1.0).unwrap();

        let stored = crate::autograd::get_gradient(&x).unwrap();
        assert_eq!(stored.data().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clipped_graph_gradient_is_what_the_step_applies() {
        let mut x = param_with_graph_gradient(3.0); // grad = [6, 6, 6]
        let mut sgd = SGD::new(0.1, None, None).with_gradient_clipping(GradientClipping::ByValue {
            min_value: -1.0,
            max_value: 1.0,
        });
        sgd.step(&mut [&mut x]).unwrap();
        // Clipped: 3.0 - 0.1 * 1.0. Unclipped it would have been 3.0 - 0.6.
        for &v in x.data().as_f32_slice().unwrap() {
            assert!((v - 2.9).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn test_gradient_helpers_see_graph_stored_gradients() {
        let x = param_with_graph_gradient(2.0);
        assert!(GradientUtils::has_gradients(&[&x]));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&[&x]), 1);
    }

    #[test]
    fn test_apply_lr_scheduler_does_not_compound_decay() {
        use crate::optim::ExponentialLR;

        // Regression: the base learning rate used to be read back from the
        // optimizer after each update, so repeated application decayed from
        // the already-decayed rate (gamma^(1+2+...+n) instead of gamma^n).
        let base_lr = 1.0;
        let scheduler = ExponentialLR::new(0.5);
        let mut sgd = SGD::new(base_lr, None, None);
        let mut param = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), true);
        param.set_grad(Some(Tensor::zeros(
            Shape::new(vec![1]),
            DataType::Float32,
            Device::cpu(),
            false,
        )));

        for expected_step in 1..=3usize {
            sgd.step(&mut [&mut param]).unwrap();
            sgd.apply_lr_scheduler(&scheduler, base_lr);
            let expected = 0.5f64.powi(expected_step as i32);
            assert!(
                (sgd.learning_rate() - expected).abs() < 1e-12,
                "step {expected_step}: got {}, want {expected}",
                sgd.learning_rate()
            );
        }
    }

    /// Run `grads` through Adagrad and return the parameter trajectory.
    fn adagrad_trajectory(
        start: f64,
        grads: &[f64],
        lr: f64,
        lr_decay: f64,
        weight_decay: f64,
        initial: f64,
    ) -> Vec<f64> {
        let mut p = Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);
        p.data_mut().as_f64_slice_mut().unwrap()[0] = start;
        let mut opt = Adagrad::new(
            lr,
            Some(lr_decay),
            Some(weight_decay),
            Some(initial),
            Some(1e-10),
        );

        let mut out = Vec::with_capacity(grads.len());
        for &g in grads {
            let mut grad =
                Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), false);
            grad.data_mut().as_f64_slice_mut().unwrap()[0] = g;
            p.set_grad(Some(grad));
            {
                let mut params = vec![&mut p];
                opt.step(&mut params).unwrap();
            }
            out.push(p.data().as_f64_slice().unwrap()[0]);
        }
        out
    }

    #[test]
    fn test_adagrad_matches_the_update_rule() {
        // Straight transcription of
        //   sum += g^2 ; clr = lr / (1 + (t-1)*lr_decay) ; p -= clr*g/(sqrt(sum)+eps)
        let grads = [0.5f64, -2.0, 1.0, 0.25];
        for &(lr, lr_decay, wd, init) in &[
            (0.1f64, 0.0f64, 0.0f64, 0.0f64),
            (0.1, 0.5, 0.0, 0.0),
            (0.1, 0.0, 0.05, 0.0),
            (0.05, 0.2, 0.01, 0.1),
        ] {
            let got = adagrad_trajectory(1.0, &grads, lr, lr_decay, wd, init);

            let eps = 1e-10;
            let mut p = 1.0f64;
            let mut sum = init;
            for (i, &g) in grads.iter().enumerate() {
                let g = g + wd * p;
                sum += g * g;
                let clr = lr / (1.0 + i as f64 * lr_decay);
                p -= clr * g / (sum.sqrt() + eps);
                assert!(
                    (got[i] - p).abs() < 1e-12,
                    "lr={lr} decay={lr_decay} wd={wd} init={init} step {i}: {} != {p}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn test_adagrad_step_decays_like_one_over_sqrt_t() {
        // Under a constant unit gradient the accumulator after t steps is t, so
        // the step is lr/(sqrt(t) + eps). RMSprop's moving average would instead
        // settle to a constant step -- this is the difference between them.
        // eps is only 1e-10 but it moves the answer by more than 1e-11, so it
        // belongs in the expected value rather than in the tolerance.
        let grads = [1.0f64; 10];
        let trajectory = adagrad_trajectory(0.0, &grads, 0.1, 0.0, 0.0, 0.0);

        let mut previous = 0.0;
        for (i, &value) in trajectory.iter().enumerate() {
            let step = previous - value;
            let expected = 0.1 / (((i + 1) as f64).sqrt() + 1e-10);
            assert!(
                (step - expected).abs() < 1e-12,
                "step {i}: {step} != {expected}"
            );
            previous = value;
        }
    }

    #[test]
    fn test_adagrad_accumulator_never_lets_the_step_grow() {
        // Gradients of wildly varying size: the accumulator only ever adds, so
        // no ordering of them can make a later step larger than an earlier one.
        let grads = [3.0f64, 0.01, -5.0, 0.001, 2.0, -0.02];
        let trajectory = adagrad_trajectory(0.0, &grads, 0.1, 0.0, 0.0, 0.0);

        let mut previous_value = 0.0;
        let mut previous_step = f64::INFINITY;
        for (i, &value) in trajectory.iter().enumerate() {
            let step = (previous_value - value).abs();
            // The step also scales with |grad|, so compare the *effective learning
            // rate* -- step divided by the gradient that produced it.
            let effective = step / grads[i].abs();
            assert!(
                effective <= previous_step + 1e-12,
                "effective lr grew at step {i}: {effective} > {previous_step}"
            );
            previous_step = effective;
            previous_value = value;
        }
    }

    #[test]
    fn test_adagrad_initial_accumulator_damps_the_first_step() {
        // Starting the accumulator at 3 makes the first denominator sqrt(1+3)=2.
        // Epsilon shifts both by ~1e-11, so carry it in the expectation.
        let eps = 1e-10;
        let plain = adagrad_trajectory(0.0, &[1.0], 0.1, 0.0, 0.0, 0.0)[0];
        let damped = adagrad_trajectory(0.0, &[1.0], 0.1, 0.0, 0.0, 3.0)[0];
        assert!((plain + 0.1 / (1.0 + eps)).abs() < 1e-15, "{plain}");
        assert!((damped + 0.1 / (2.0 + eps)).abs() < 1e-15, "{damped}");
    }

    #[test]
    fn test_adagrad_creation_and_defaults() {
        let adagrad = Adagrad::new(0.01, None, None, None, None);
        assert_eq!(adagrad.learning_rate(), 0.01);
        assert_eq!(adagrad.lr_decay(), 0.0);
        assert_eq!(adagrad.weight_decay(), 0.0);
        assert_eq!(adagrad.initial_accumulator_value(), 0.0);
        // Adagrad floors a sum that only grows, so its epsilon is smaller than
        // the 1e-8 the moving-average optimizers use.
        assert_eq!(adagrad.epsilon(), 1e-10);
        assert_eq!(adagrad.step_count(), 0);
    }

    #[test]
    fn test_adagrad_rejects_non_float_parameters() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), true);
        let grad = Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), false);
        p.set_grad(Some(grad));
        let mut opt = Adagrad::new(0.1, None, None, None, None);
        let mut params = vec![&mut p];
        assert!(opt.step(&mut params).is_err());
    }

    #[test]
    fn test_nadam_matches_the_published_update() {
        // Direct transcription of the NAdam recurrence, including the running
        // product of the momentum schedule.
        let grads = [0.5f64, -2.0, 1.0, 0.25];
        let (lr, b1, b2, eps, psi) = (0.01f64, 0.9f64, 0.999f64, 1e-8f64, 0.004f64);

        let mut p = Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);
        p.data_mut().as_f64_slice_mut().unwrap()[0] = 1.0;
        let mut opt = NAdam::new(lr, Some(b1), Some(b2), Some(eps), Some(0.0), Some(psi));

        let (mut m, mut v, mut mu_product) = (0.0f64, 0.0f64, 1.0f64);
        let mut expected = 1.0f64;

        for (i, &g) in grads.iter().enumerate() {
            let t = (i + 1) as f64;
            let mu = b1 * (1.0 - 0.5 * 0.96f64.powf(t * psi));
            let mu_next = b1 * (1.0 - 0.5 * 0.96f64.powf((t + 1.0) * psi));
            mu_product *= mu;
            let mu_product_next = mu_product * mu_next;

            m = b1 * m + (1.0 - b1) * g;
            v = b2 * v + (1.0 - b2) * g * g;
            let denom = (v / (1.0 - b2.powf(t))).sqrt() + eps;
            expected -= lr
                * ((1.0 - mu) / (1.0 - mu_product) * g + mu_next / (1.0 - mu_product_next) * m)
                / denom;

            let mut grad =
                Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), false);
            grad.data_mut().as_f64_slice_mut().unwrap()[0] = g;
            p.set_grad(Some(grad));
            {
                let mut params = vec![&mut p];
                opt.step(&mut params).unwrap();
            }
            let got = p.data().as_f64_slice().unwrap()[0];
            assert!(
                (got - expected).abs() < 1e-12,
                "step {i}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn test_nadam_schedule_advances_once_per_step_not_per_parameter() {
        // The momentum product is shared across parameters, so it must be
        // advanced in `step` rather than inside the per-parameter loop. Two
        // parameters with the same state and gradient must stay identical; a
        // per-parameter advance would put the second one a step ahead.
        let make = || {
            let mut t = Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), true);
            t.data_mut().as_f64_slice_mut().unwrap()[0] = 1.0;
            t
        };
        let (mut a, mut b) = (make(), make());
        let mut opt = NAdam::new(0.1, None, None, None, None, None);

        for _ in 0..5 {
            for t in [&mut a, &mut b] {
                let mut grad =
                    Tensor::zeros(Shape::new(vec![1]), DataType::Float64, Device::cpu(), false);
                grad.data_mut().as_f64_slice_mut().unwrap()[0] = 1.0;
                t.set_grad(Some(grad));
            }
            let mut params = vec![&mut a, &mut b];
            opt.step(&mut params).unwrap();
        }

        let (va, vb) = (
            a.data().as_f64_slice().unwrap()[0],
            b.data().as_f64_slice().unwrap()[0],
        );
        assert_eq!(va, vb, "parameters diverged: {va} vs {vb}");
    }

    #[test]
    fn test_nadam_creation_and_defaults() {
        let nadam = NAdam::new(0.002, None, None, None, None, None);
        assert_eq!(nadam.learning_rate(), 0.002);
        assert_eq!(nadam.beta1(), 0.9);
        assert_eq!(nadam.beta2(), 0.999);
        assert_eq!(nadam.epsilon(), 1e-8);
        assert_eq!(nadam.weight_decay(), 0.0);
        assert_eq!(nadam.momentum_decay(), 0.004);
        assert_eq!(nadam.step_count(), 0);
    }

    #[test]
    fn test_nadam_rejects_non_float_parameters() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), true);
        let grad = Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), false);
        p.set_grad(Some(grad));
        let mut opt = NAdam::new(0.01, None, None, None, None, None);
        let mut params = vec![&mut p];
        assert!(opt.step(&mut params).is_err());
    }
}
