// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::LearningRateScheduler;
use crate::{error::Result, tensor::Tensor};
use rayon::prelude::*;
use smallvec::SmallVec;

/// Utility functions for gradient operations
pub struct GradientUtils;

impl GradientUtils {
    /// Compute the L2 norm of gradients across all parameters
    pub fn compute_grad_norm(parameters: &[&Tensor]) -> Result<f64> {
        let total_sq_norm: f64 = parameters
            .par_iter()
            .map(|param| {
                if let Some(grad) = param.grad() {
                    match grad.dtype() {
                        crate::tensor::DataType::Float32 => grad
                            .data()
                            .as_f32_slice()
                            .unwrap()
                            .par_iter()
                            .map(|&v| (v as f64) * (v as f64))
                            .sum::<f64>(),
                        crate::tensor::DataType::Float64 => grad
                            .data()
                            .as_f64_slice()
                            .unwrap()
                            .par_iter()
                            .map(|&v| v * v)
                            .sum::<f64>(),
                        _ => 0.0,
                    }
                } else {
                    0.0
                }
            })
            .sum();
        Ok(total_sq_norm.sqrt())
    }

    /// Apply gradient clipping by norm to a set of parameters
    pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f64) -> Result<f64> {
        let mut refs: SmallVec<[&Tensor; 16]> = SmallVec::with_capacity(parameters.len());
        for p in parameters.iter() {
            refs.push(&**p);
        }
        let total_norm = Self::compute_grad_norm(&refs)?;
        drop(refs);
        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);
            parameters.par_iter_mut().for_each(|param| {
                if let Some(grad) = param.grad_mut() {
                    match grad.dtype() {
                        crate::tensor::DataType::Float32 => {
                            let g = grad.data_mut().as_f32_slice_mut().unwrap();
                            let coef = clip_coef as f32;
                            g.par_iter_mut().for_each(|v| *v *= coef);
                        }
                        crate::tensor::DataType::Float64 => {
                            let g = grad.data_mut().as_f64_slice_mut().unwrap();
                            g.par_iter_mut().for_each(|v| *v *= clip_coef);
                        }
                        _ => {}
                    }
                }
            });
        }

        Ok(total_norm)
    }

    /// Apply gradient clipping by value to a set of parameters
    pub fn clip_grad_value(
        parameters: &mut [&mut Tensor],
        min_value: f64,
        max_value: f64,
    ) -> Result<()> {
        parameters.par_iter_mut().for_each(|param| {
            if let Some(grad) = param.grad_mut() {
                match grad.dtype() {
                    crate::tensor::DataType::Float32 => {
                        let g = grad.data_mut().as_f32_slice_mut().unwrap();
                        let min = min_value as f32;
                        let max = max_value as f32;
                        g.par_iter_mut().for_each(|v| {
                            *v = v.clamp(min, max);
                        });
                    }
                    crate::tensor::DataType::Float64 => {
                        let g = grad.data_mut().as_f64_slice_mut().unwrap();
                        g.par_iter_mut().for_each(|v| {
                            *v = v.clamp(min_value, max_value);
                        });
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }

    /// Check if any parameters have gradients
    pub fn has_gradients(parameters: &[&Tensor]) -> bool {
        parameters.iter().any(|param| param.grad().is_some())
    }

    /// Count the number of parameters with gradients
    pub fn count_parameters_with_gradients(parameters: &[&Tensor]) -> usize {
        parameters
            .iter()
            .filter(|param| param.grad().is_some())
            .count()
    }
}

/// Learning rate scheduler utilities
pub struct SchedulerUtils;

impl SchedulerUtils {
    /// Create a linear warmup scheduler that increases learning rate linearly
    pub fn linear_warmup(warmup_steps: usize) -> LinearWarmupScheduler {
        LinearWarmupScheduler::new(warmup_steps)
    }

    /// Create a polynomial decay scheduler
    pub fn polynomial_decay(
        decay_steps: usize,
        end_lr: f64,
        power: f64,
    ) -> PolynomialDecayScheduler {
        PolynomialDecayScheduler::new(decay_steps, end_lr, power)
    }

    /// Create a multi-step scheduler with multiple decay points
    pub fn multi_step(milestones: Vec<usize>, gamma: f64) -> MultiStepScheduler {
        MultiStepScheduler::new(milestones, gamma)
    }
}

/// Linear warmup learning rate scheduler
#[derive(Debug, Clone)]
pub struct LinearWarmupScheduler {
    warmup_steps: usize,
}

impl LinearWarmupScheduler {
    pub fn new(warmup_steps: usize) -> Self {
        Self { warmup_steps }
    }
}

impl LearningRateScheduler for LinearWarmupScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if step < self.warmup_steps {
            base_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            base_lr
        }
    }
}

/// Polynomial decay learning rate scheduler
#[derive(Debug, Clone)]
pub struct PolynomialDecayScheduler {
    decay_steps: usize,
    end_lr: f64,
    power: f64,
}

impl PolynomialDecayScheduler {
    pub fn new(decay_steps: usize, end_lr: f64, power: f64) -> Self {
        Self {
            decay_steps,
            end_lr,
            power,
        }
    }
}

impl LearningRateScheduler for PolynomialDecayScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if step >= self.decay_steps {
            return self.end_lr;
        }

        let decay_factor = (1.0 - step as f64 / self.decay_steps as f64).powf(self.power);
        (base_lr - self.end_lr) * decay_factor + self.end_lr
    }
}

/// Multi-step learning rate scheduler
#[derive(Debug, Clone)]
pub struct MultiStepScheduler {
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepScheduler {
    pub fn new(mut milestones: Vec<usize>, gamma: f64) -> Self {
        milestones.sort_unstable();
        Self { milestones, gamma }
    }
}

impl LearningRateScheduler for MultiStepScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        let decay_count = self
            .milestones
            .iter()
            .filter(|&&milestone| step >= milestone)
            .count();
        base_lr * self.gamma.powi(decay_count as i32)
    }
}

/// Composite scheduler that combines multiple schedulers
pub struct CompositeScheduler {
    schedulers: Vec<(Box<dyn LearningRateScheduler>, usize)>, // (scheduler, start_step)
}

impl CompositeScheduler {
    pub fn new() -> Self {
        Self {
            schedulers: Vec::new(),
        }
    }

    /// Add a scheduler that starts at a specific step
    pub fn add_scheduler(
        mut self,
        scheduler: Box<dyn LearningRateScheduler>,
        start_step: usize,
    ) -> Self {
        self.schedulers.push((scheduler, start_step));
        // Sort by start step
        self.schedulers.sort_by_key(|(_, start)| *start);
        self
    }
}

impl Default for CompositeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningRateScheduler for CompositeScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        // Find the most recent scheduler that should be active
        let mut current_lr = base_lr;

        for (scheduler, start_step) in &self.schedulers {
            if step >= *start_step {
                current_lr = scheduler.get_lr(step - start_step, base_lr);
            } else {
                break;
            }
        }

        current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor},
    };

    #[test]
    fn test_gradient_utils_has_gradients() {
        let shape = Shape::new(vec![2, 2]);
        let tensor1 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let mut tensor2 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        // Set gradient on tensor2
        let grad = Tensor::ones(shape, DataType::Float32, Device::cpu(), false);
        tensor2.set_grad(Some(grad));

        let params = vec![&tensor1, &tensor2];
        assert!(GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 1);
    }

    #[test]
    fn test_gradient_utils_no_gradients() {
        let shape = Shape::new(vec![2, 2]);
        let tensor1 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let tensor2 = Tensor::zeros(shape, DataType::Float64, Device::cpu(), true);

        let params = vec![&tensor1, &tensor2];
        assert!(!GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 0);
        assert_eq!(GradientUtils::compute_grad_norm(&params).unwrap(), 0.0);
    }

    #[test]
    fn test_clip_grad_norm_clips_float64_and_skips_non_float() {
        let mut float_param =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut int_param =
            Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), true);

        let mut float_grad =
            Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        float_grad
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[6.0, 8.0]);
        let mut int_grad = Tensor::ones(Shape::new(vec![2]), DataType::Int64, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i64_slice_mut()
            .unwrap()
            .copy_from_slice(&[7, -3]);

        float_param.set_grad(Some(float_grad));
        int_param.set_grad(Some(int_grad));

        let mut params = vec![&mut float_param, &mut int_param];
        let original_norm = GradientUtils::clip_grad_norm(&mut params, 5.0).unwrap();
        assert!((original_norm - 10.0).abs() < 1e-10);

        let scaled_float = params[0].grad().unwrap().data().as_f64_slice().unwrap();
        assert!((scaled_float[0] - 3.0).abs() < 1e-6);
        assert!((scaled_float[1] - 4.0).abs() < 1e-6);

        let unchanged_int = params[1].grad().unwrap().data().as_i64_slice().unwrap();
        assert_eq!(unchanged_int, &[7, -3]);
    }

    #[test]
    fn test_clip_grad_value_no_gradients_is_noop() {
        let mut p1 = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut p2 = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);

        let mut params = vec![&mut p1, &mut p2];
        GradientUtils::clip_grad_value(&mut params, -1.0, 1.0).unwrap();

        assert!(params[0].grad().is_none());
        assert!(params[1].grad().is_none());
    }

    #[test]
    fn test_clip_grad_norm_empty_parameter_list() {
        let mut params: Vec<&mut Tensor> = Vec::new();
        let norm = GradientUtils::clip_grad_norm(&mut params, 1.0).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_value_empty_parameter_list() {
        let mut params: Vec<&mut Tensor> = Vec::new();
        GradientUtils::clip_grad_value(&mut params, -0.5, 0.5).unwrap();
    }
    #[test]
    fn test_linear_warmup_zero_steps_returns_base_lr() {
        let scheduler = LinearWarmupScheduler::new(0);
        assert_eq!(scheduler.get_lr(0, 0.2), 0.2);
        assert_eq!(scheduler.get_lr(10, 0.2), 0.2);
    }

    #[test]
    fn test_polynomial_decay_zero_decay_steps_returns_end_lr() {
        let scheduler = PolynomialDecayScheduler::new(0, 0.05, 2.0);
        assert_eq!(scheduler.get_lr(0, 0.5), 0.05);
        assert_eq!(scheduler.get_lr(5, 0.5), 0.05);
    }

    #[test]
    fn test_composite_scheduler_sorts_start_steps() {
        let base_lr = 1.0;
        let scheduler = CompositeScheduler::new()
            .add_scheduler(Box::new(MultiStepScheduler::new(vec![1], 0.5)), 10)
            .add_scheduler(Box::new(LinearWarmupScheduler::new(5)), 0);

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(4, base_lr), 0.8);
        assert_eq!(scheduler.get_lr(10, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(11, base_lr), 0.5);
    }

    #[test]
    fn test_linear_warmup_scheduler() {
        let scheduler = LinearWarmupScheduler::new(10);
        let base_lr = 0.1;

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(5, base_lr), 0.05);
        assert_eq!(scheduler.get_lr(10, base_lr), 0.1);
        assert_eq!(scheduler.get_lr(15, base_lr), 0.1);
    }

    #[test]
    fn test_polynomial_decay_scheduler() {
        let scheduler = PolynomialDecayScheduler::new(100, 0.01, 2.0);
        let base_lr = 0.1;

        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert!(scheduler.get_lr(50, base_lr) > 0.01);
        assert!(scheduler.get_lr(50, base_lr) < base_lr);
        assert_eq!(scheduler.get_lr(100, base_lr), 0.01);
        assert_eq!(scheduler.get_lr(150, base_lr), 0.01);
    }

    #[test]
    fn test_multi_step_scheduler() {
        let scheduler = MultiStepScheduler::new(vec![30, 60, 90], 0.1);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(29, base_lr), 1.0);
        assert!((scheduler.get_lr(30, base_lr) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(60, base_lr) - 0.01).abs() < 1e-10);
        assert!((scheduler.get_lr(90, base_lr) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_step_lr_scheduler() {
        use super::super::optimizer::StepLR;
        let scheduler = StepLR::new(10, 0.5);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(9, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(10, base_lr), 0.5);
        assert_eq!(scheduler.get_lr(20, base_lr), 0.25);
    }

    #[test]
    fn test_step_lr_zero_step_size_defaults() {
        use super::super::optimizer::StepLR;
        let scheduler = StepLR::new(0, 0.5);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 1.0);
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        use super::super::optimizer::ExponentialLR;
        let scheduler = ExponentialLR::new(0.9);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 0.9);
        assert!((scheduler.get_lr(2, base_lr) - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        use super::super::optimizer::CosineAnnealingLR;
        let scheduler = CosineAnnealingLR::new(100, 0.0);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        // At step 50 of 100, we have cos(π * 50/100) = cos(π/2) = 0
        // So lr = 0 + (1-0) * (1+0)/2 = 0.5
        assert!((scheduler.get_lr(50, base_lr) - 0.5).abs() < 1e-10);
        // At step 100, lr should reach eta_min.
        assert_eq!(scheduler.get_lr(100, base_lr), 0.0);
        // After t_max, lr should stay at eta_min.
        assert_eq!(scheduler.get_lr(150, base_lr), 0.0);
    }

    #[test]
    fn test_cosine_annealing_zero_t_max_defaults() {
        use super::super::optimizer::CosineAnnealingLR;
        let scheduler = CosineAnnealingLR::new(0, 0.1);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 1.0);
    }

    #[test]
    fn test_compute_grad_norm_float32_and_float64() {
        let shape = Shape::new(vec![2]);
        let mut float32_param =
            Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let mut float64_param = Tensor::zeros(shape, DataType::Float64, Device::cpu(), true);

        let mut grad_f32 =
            Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad_f32
            .data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        let mut grad_f64 =
            Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        grad_f64
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[12.0, 0.0]);

        float32_param.set_grad(Some(grad_f32));
        float64_param.set_grad(Some(grad_f64));

        let params = vec![&float32_param, &float64_param];
        let norm = GradientUtils::compute_grad_norm(&params).unwrap();
        assert!((norm - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_grad_norm_ignores_non_float_gradients() {
        let mut param = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut int_grad = Tensor::ones(Shape::new(vec![2]), DataType::Int32, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&[10, -10]);
        param.set_grad(Some(int_grad));

        let params = vec![&param];
        let norm = GradientUtils::compute_grad_norm(&params).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_norm_scales_down_gradients() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        p.set_grad(Some(grad));

        let mut params = vec![&mut p];
        let original_norm = GradientUtils::clip_grad_norm(&mut params, 2.5).unwrap();
        assert!((original_norm - 5.0).abs() < 1e-10);

        let scaled = params[0].grad().unwrap().data().as_f32_slice().unwrap();
        assert!((scaled[0] - 1.5).abs() < 1e-3);
        assert!((scaled[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_clip_grad_norm_handles_parameters_without_gradients_in_clip_pass() {
        let mut with_grad =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut without_grad =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);

        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[10.0, 0.0]);
        with_grad.set_grad(Some(grad));

        let mut params = vec![&mut with_grad, &mut without_grad];
        let norm = GradientUtils::clip_grad_norm(&mut params, 1.0).unwrap();
        assert!(norm > 1.0);
        assert!(params[1].grad().is_none());
    }
    #[test]
    fn test_clip_grad_norm_noop_when_within_threshold() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        grad.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[1.0, 2.0]);
        p.set_grad(Some(grad));

        let before = p.grad().unwrap().data().as_f64_slice().unwrap().to_vec();
        let mut params = vec![&mut p];
        let norm = GradientUtils::clip_grad_norm(&mut params, 10.0).unwrap();
        assert!((norm - (5.0f64).sqrt()).abs() < 1e-10);

        let after = params[0].grad().unwrap().data().as_f64_slice().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn test_clip_grad_value_for_float_and_non_float() {
        let mut f32_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let mut f64_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Float64, Device::cpu(), true);
        let mut i32_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Int32, Device::cpu(), true);

        let mut grad_f32 =
            Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), false);
        grad_f32
            .data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[-3.0, 0.5, 9.0]);
        let mut grad_f64 =
            Tensor::ones(Shape::new(vec![3]), DataType::Float64, Device::cpu(), false);
        grad_f64
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[-2.0, 0.1, 8.0]);
        let mut grad_i32 = Tensor::ones(Shape::new(vec![3]), DataType::Int32, Device::cpu(), false);
        grad_i32
            .data_mut()
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&[-2, 5, 10]);

        f32_param.set_grad(Some(grad_f32));
        f64_param.set_grad(Some(grad_f64));
        i32_param.set_grad(Some(grad_i32));

        let mut params = vec![&mut f32_param, &mut f64_param, &mut i32_param];
        GradientUtils::clip_grad_value(&mut params, -1.0, 1.0).unwrap();

        let f32_data = params[0].grad().unwrap().data().as_f32_slice().unwrap();
        assert_eq!(f32_data, &[-1.0, 0.5, 1.0]);

        let f64_data = params[1].grad().unwrap().data().as_f64_slice().unwrap();
        assert_eq!(f64_data, &[-1.0, 0.1, 1.0]);

        let i32_data = params[2].grad().unwrap().data().as_i32_slice().unwrap();
        assert_eq!(i32_data, &[-2, 5, 10]);
    }

    #[test]
    fn test_scheduler_utils_factory_methods() {
        let linear = SchedulerUtils::linear_warmup(4);
        assert_eq!(linear.get_lr(2, 0.4), 0.2);

        let poly = SchedulerUtils::polynomial_decay(4, 0.1, 1.0);
        assert_eq!(poly.get_lr(4, 0.5), 0.1);

        let multistep = SchedulerUtils::multi_step(vec![4, 2], 0.1);
        assert!((multistep.get_lr(4, 1.0) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_composite_scheduler_default_and_switching() {
        let base_lr = 1.0;
        let scheduler = CompositeScheduler::default()
            .add_scheduler(Box::new(LinearWarmupScheduler::new(2)), 0)
            .add_scheduler(Box::new(MultiStepScheduler::new(vec![1], 0.1)), 5);

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(2, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(5, base_lr), 1.0);
        assert!((scheduler.get_lr(6, base_lr) - 0.1).abs() < 1e-12);

        let empty = CompositeScheduler::new();
        assert_eq!(empty.get_lr(10, base_lr), base_lr);
    }
}
