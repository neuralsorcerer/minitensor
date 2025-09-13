// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::utils::GradientUtils;
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use rustc_hash::FxHashMap;

/// Parameter group for managing different learning rates and settings
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Parameters in this group
    pub params: Vec<TensorId>,
    /// Learning rate for this group
    pub lr: f64,
    /// Weight decay for this group
    pub weight_decay: f64,
    /// Additional group-specific options
    pub options: FxHashMap<String, f64>,
}

impl ParameterGroup {
    /// Create a new parameter group
    pub fn new(params: Vec<TensorId>, lr: f64) -> Self {
        Self {
            params,
            lr,
            weight_decay: 0.0,
            options: FxHashMap::default(),
        }
    }

    /// Create a parameter group with weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Add a custom option to the parameter group
    pub fn with_option(mut self, key: String, value: f64) -> Self {
        self.options.insert(key, value);
        self
    }

    /// Get an option value
    pub fn get_option(&self, key: &str) -> Option<f64> {
        self.options.get(key).copied()
    }
}

/// Gradient clipping configuration
#[derive(Debug, Clone)]
pub enum GradientClipping {
    /// No gradient clipping
    None,
    /// Clip gradients by norm
    ByNorm { max_norm: f64 },
    /// Clip gradients by value
    ByValue { min_value: f64, max_value: f64 },
}

impl Default for GradientClipping {
    fn default() -> Self {
        Self::None
    }
}

/// Learning rate scheduler interface
pub trait LearningRateScheduler: Send + Sync {
    /// Get the learning rate for the current step
    fn get_lr(&self, step: usize, base_lr: f64) -> f64;

    /// Update scheduler state (if needed)
    fn step(&mut self) {}
}

/// Constant learning rate scheduler
#[derive(Debug, Clone)]
pub struct ConstantLR;

impl LearningRateScheduler for ConstantLR {
    fn get_lr(&self, _step: usize, base_lr: f64) -> f64 {
        base_lr
    }
}

/// Step learning rate scheduler
#[derive(Debug, Clone)]
pub struct StepLR {
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f64) -> Self {
        Self { step_size, gamma }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        let decay_factor = self.gamma.powi((step / self.step_size) as i32);
        base_lr * decay_factor
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        base_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f64,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f64) -> Self {
        Self { t_max, eta_min }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if step == 0 {
            return base_lr;
        }

        let t = (step % self.t_max) as f64;
        let t_max = self.t_max as f64;

        // Standard cosine annealing formula
        // At t=0: cos(0) = 1, lr = base_lr
        // At t=t_max/2: cos(π/2) = 0, lr = (base_lr + eta_min)/2
        // At t=t_max: cos(π) = -1, lr = eta_min
        self.eta_min
            + (base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * t / t_max).cos()) / 2.0
    }
}

/// Trait for optimization algorithms
pub trait Optimizer: Send + Sync {
    /// Perform one optimization step
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;

    /// Zero out gradients of parameters
    fn zero_grad(&self, parameters: &mut [&mut Tensor], set_to_none: bool) -> Result<()>;

    /// Get the learning rate (for single parameter group optimizers)
    fn learning_rate(&self) -> f64;

    /// Set the learning rate (for single parameter group optimizers)
    fn set_learning_rate(&mut self, lr: f64);

    /// Get parameter groups
    fn param_groups(&self) -> &[ParameterGroup] {
        // Default implementation for backward compatibility
        &[]
    }

    /// Get mutable parameter groups
    fn param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        // Default implementation for backward compatibility
        &mut []
    }

    /// Add a parameter group
    fn add_param_group(&mut self, _group: ParameterGroup) -> Result<()> {
        // Default implementation for backward compatibility
        Ok(())
    }

    /// Get the current step count
    fn step_count(&self) -> usize {
        0
    }

    /// Apply gradient clipping to parameters
    fn clip_gradients(
        &self,
        parameters: &mut [&mut Tensor],
        clipping: &GradientClipping,
    ) -> Result<()> {
        match clipping {
            GradientClipping::None => Ok(()),
            GradientClipping::ByNorm { max_norm } => self.clip_grad_norm(parameters, *max_norm),
            GradientClipping::ByValue {
                min_value,
                max_value,
            } => self.clip_grad_value(parameters, *min_value, *max_value),
        }
    }

    /// Clip gradients by norm
    fn clip_grad_norm(&self, parameters: &mut [&mut Tensor], max_norm: f64) -> Result<()> {
        GradientUtils::clip_grad_norm(parameters, max_norm).map(|_| ())
    }

    /// Clip gradients by value
    fn clip_grad_value(
        &self,
        parameters: &mut [&mut Tensor],
        min_value: f64,
        max_value: f64,
    ) -> Result<()> {
        GradientUtils::clip_grad_value(parameters, min_value, max_value)
    }

    /// Apply learning rate scheduling
    fn apply_lr_scheduler(&mut self, scheduler: &dyn LearningRateScheduler) {
        let step = self.step_count();

        // For single parameter group optimizers
        if self.param_groups().is_empty() {
            let new_lr = scheduler.get_lr(step, self.learning_rate());
            self.set_learning_rate(new_lr);
        } else {
            // For multi parameter group optimizers
            for group in self.param_groups_mut() {
                let new_lr = scheduler.get_lr(step, group.lr);
                group.lr = new_lr;
            }
        }
    }
}
