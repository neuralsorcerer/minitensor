// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(test)]
mod tests {
    use crate::optim::{Adam, Optimizer, ParameterGroup, RMSprop, SGD};
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
        sgd.zero_grad(&mut params).unwrap();

        assert!(!tensor1.has_grad());
        assert!(!tensor2.has_grad());
    }

    #[test]
    fn test_optimizer_step_not_implemented() {
        let mut sgd = SGD::new(0.01, None, None);
        let mut tensor = Tensor::zeros(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );

        // Set a gradient
        let grad = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        tensor.set_grad(Some(grad));

        let mut params = vec![&mut tensor];
        let result = sgd.step(&mut params);

        // Should return not implemented error for now
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not yet implemented"));
        }
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
}
