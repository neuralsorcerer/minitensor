// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{GradientClipping, Optimizer, ParameterGroup};
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use std::collections::HashMap;

/// RMSprop optimizer with parameter groups
pub struct RMSprop {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Default learning rate (for backward compatibility)
    default_lr: f64,
    /// Alpha coefficient for moving average
    alpha: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay coefficient
    weight_decay: f64,
    /// Momentum coefficient
    momentum: f64,
    /// Whether to use centered variant
    centered: bool,
    /// Square average buffers
    square_avg: HashMap<TensorId, Tensor>,
    /// Momentum buffers
    momentum_buffer: HashMap<TensorId, Tensor>,
    /// Gradient average buffers (for centered variant)
    grad_avg: HashMap<TensorId, Tensor>,
    /// Current step count
    step_count: usize,
    /// Gradient clipping configuration
    gradient_clipping: GradientClipping,
}

impl RMSprop {
    /// Create a new RMSprop optimizer with single parameter group
    pub fn new(
        learning_rate: f64,
        alpha: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
        momentum: Option<f64>,
    ) -> Self {
        Self {
            param_groups: Vec::new(),
            default_lr: learning_rate,
            alpha: alpha.unwrap_or(0.99),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            momentum: momentum.unwrap_or(0.0),
            centered: false,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new RMSprop optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        alpha: f64,
        epsilon: f64,
        momentum: f64,
    ) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        Self {
            param_groups,
            default_lr,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum,
            centered: false,
            square_avg: HashMap::new(),
            momentum_buffer: HashMap::new(),
            grad_avg: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Enable centered variant
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get alpha coefficient
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set alpha coefficient
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Set epsilon value
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Get weight decay coefficient
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.weight_decay = weight_decay;
    }

    /// Get momentum coefficient
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Set momentum coefficient
    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }

    /// Check if using centered variant
    pub fn is_centered(&self) -> bool {
        self.centered
    }

    /// Get learning rate for a specific parameter
    fn get_param_lr(&self, param_id: TensorId) -> f64 {
        // Find parameter group containing this parameter
        for group in &self.param_groups {
            if group.params.contains(&param_id) {
                return group.lr;
            }
        }
        // Default to global learning rate
        self.default_lr
    }

    /// Get weight decay for a specific parameter
    fn get_param_weight_decay(&self, param_id: TensorId) -> f64 {
        // Find parameter group containing this parameter
        for group in &self.param_groups {
            if group.params.contains(&param_id) {
                return group.weight_decay;
            }
        }
        // Default to global weight decay
        self.weight_decay
    }

    /// Apply weight decay to gradient
    fn apply_weight_decay(
        &self,
        _param: &Tensor,
        grad: &Tensor,
        _weight_decay: f64,
    ) -> Result<Tensor> {
        // For now, return the original gradient
        // In a full implementation, we would compute: grad + weight_decay * _param
        Ok(grad.clone())
    }

    /// Apply RMSprop optimization update
    fn apply_rmsprop_update(&mut self, param: &mut Tensor, grad: &Tensor, lr: f64) -> Result<()> {
        let param_id = param.id();

        // Get or create square average buffer
        let square_avg = if let Some(avg) = self.square_avg.get(&param_id) {
            avg.clone()
        } else {
            let avg = Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
            self.square_avg.insert(param_id, avg.clone());
            avg
        };

        // Get or create momentum buffer if momentum > 0
        let momentum_buffer = if self.momentum > 0.0 {
            if let Some(buf) = self.momentum_buffer.get(&param_id) {
                buf.clone()
            } else {
                let buf =
                    Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
                self.momentum_buffer.insert(param_id, buf.clone());
                buf
            }
        } else {
            // Dummy tensor for no momentum case
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        };

        // Get or create gradient average buffer for centered variant
        let grad_avg = if self.centered {
            if let Some(avg) = self.grad_avg.get(&param_id) {
                avg.clone()
            } else {
                let avg =
                    Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
                self.grad_avg.insert(param_id, avg.clone());
                avg
            }
        } else {
            // Dummy tensor for non-centered case
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        };

        // Perform RMSprop update
        self.rmsprop_step(param, grad, &square_avg, &momentum_buffer, &grad_avg, lr)?;

        Ok(())
    }

    /// Perform the actual RMSprop optimization step
    fn rmsprop_step(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        _square_avg: &Tensor,
        _momentum_buffer: &Tensor,
        _grad_avg: &Tensor,
        _lr: f64,
    ) -> Result<()> {
        if param.device() != grad.device() {
            return Err(crate::error::MinitensorError::device_mismatch(
                param.device().to_string(),
                grad.device().to_string(),
            ));
        }

        if param.shape() != grad.shape() {
            return Err(crate::error::MinitensorError::shape_mismatch(
                param.shape().dims().to_vec(),
                grad.shape().dims().to_vec(),
            ));
        }

        // For now, we'll implement a simplified version that doesn't require mutable access to Arc
        // In a full implementation, we would need to handle this differently
        Err(crate::error::MinitensorError::not_implemented(
            "Mutable tensor updates not yet implemented - requires tensor arithmetic operations",
        ))
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        // Apply gradient clipping if configured
        self.clip_gradients(parameters, &self.gradient_clipping)?;

        // Increment step count
        self.step_count += 1;

        // Process each parameter
        for param in parameters.iter_mut() {
            if !param.requires_grad() {
                continue;
            }

            let grad = match param.grad() {
                Some(g) => g,
                None => continue, // Skip parameters without gradients
            };

            // Get learning rate for this parameter
            let lr = self.get_param_lr(param.id());
            let weight_decay = self.get_param_weight_decay(param.id());

            // Apply weight decay if configured
            let effective_grad = if weight_decay > 0.0 {
                self.apply_weight_decay(param, grad, weight_decay)?
            } else {
                grad.as_ref().clone()
            };

            // Apply RMSprop update
            self.apply_rmsprop_update(param, &effective_grad, lr)?;
        }

        Ok(())
    }

    fn zero_grad(&self, parameters: &mut [&mut Tensor]) -> Result<()> {
        for param in parameters.iter_mut() {
            param.zero_grad();
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.default_lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.default_lr = lr;
        // Also update all parameter groups if they exist
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn param_groups(&self) -> &[ParameterGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, group: ParameterGroup) -> Result<()> {
        self.param_groups.push(group);
        Ok(())
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}
