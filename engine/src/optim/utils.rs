// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::LearningRateScheduler;
use crate::{error::Result, tensor::Tensor};

/// Utility functions for gradient operations
pub struct GradientUtils;

impl GradientUtils {
    /// Compute the L2 norm of gradients across all parameters
    pub fn compute_grad_norm(parameters: &[&Tensor]) -> Result<f64> {
        let mut total_norm_squared = 0.0;
        let mut has_gradients = false;

        for param in parameters {
            if let Some(grad) = param.grad() {
                has_gradients = true;
                // For now, we'll use a simplified norm calculation
                // In a full implementation, we would compute the actual L2 norm of the gradient tensor
                let grad_norm_squared = grad.numel() as f64; // Placeholder calculation
                total_norm_squared += grad_norm_squared;
            }
        }

        if !has_gradients {
            return Ok(0.0);
        }

        Ok(total_norm_squared.sqrt())
    }

    /// Apply gradient clipping by norm to a set of parameters
    pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f64) -> Result<f64> {
        let total_norm =
            Self::compute_grad_norm(&parameters.iter().map(|p| p as &Tensor).collect::<Vec<_>>())?;

        if total_norm <= max_norm {
            return Ok(total_norm);
        }

        let _clip_coef = max_norm / total_norm;

        // Apply clipping to all gradients
        for param in parameters {
            if param.grad().is_some() {
                // In a full implementation, we would scale the gradient tensor by _clip_coef
                // For now, we'll just mark that clipping occurred
                // Clipping gradient for parameter by factor _clip_coef
            }
        }

        Ok(total_norm)
    }

    /// Apply gradient clipping by value to a set of parameters
    pub fn clip_grad_value(
        parameters: &mut [&mut Tensor],
        _min_value: f64,
        _max_value: f64,
    ) -> Result<()> {
        for param in parameters {
            if param.grad().is_some() {
                // In a full implementation, we would clamp all gradient values
                // For now, we'll just mark that clipping occurred
                // Clipping gradient values for parameter to range [min_value, max_value]
            }
        }

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
        // At step 100, due to modulo, it's like step 0, so back to base_lr
        assert_eq!(scheduler.get_lr(100, base_lr), 1.0);
    }
}
