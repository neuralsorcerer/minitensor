// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{GradientClipping, Optimizer, ParameterGroup};
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use std::collections::HashMap;

/// SGD optimizer with momentum support and parameter groups
pub struct SGD {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Default learning rate (for backward compatibility)
    default_lr: f64,
    /// Momentum coefficient
    momentum: f64,
    /// Weight decay coefficient
    weight_decay: f64,
    /// Dampening for momentum
    dampening: f64,
    /// Whether to use Nesterov momentum
    nesterov: bool,
    /// Velocity buffers for momentum
    velocity: HashMap<TensorId, Tensor>,
    /// Current step count
    step_count: usize,
    /// Gradient clipping configuration
    gradient_clipping: GradientClipping,
}

impl SGD {
    /// Create a new SGD optimizer with single parameter group
    pub fn new(learning_rate: f64, momentum: Option<f64>, weight_decay: Option<f64>) -> Self {
        Self {
            param_groups: Vec::new(),
            default_lr: learning_rate,
            momentum: momentum.unwrap_or(0.0),
            weight_decay: weight_decay.unwrap_or(0.0),
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new SGD optimizer with parameter groups
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, momentum: f64) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        Self {
            param_groups,
            default_lr,
            momentum,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Set dampening for momentum
    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Enable Nesterov momentum
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get momentum coefficient
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Set momentum coefficient
    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }

    /// Get weight decay coefficient
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.weight_decay = weight_decay;
    }

    /// Check if using Nesterov momentum
    pub fn is_nesterov(&self) -> bool {
        self.nesterov
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
        // This requires tensor addition which will be implemented in arithmetic operations
        Ok(grad.clone())
    }

    /// Apply simple SGD update without momentum
    fn apply_simple_update(&mut self, param: &mut Tensor, grad: &Tensor, _lr: f64) -> Result<()> {
        // Simple update: param = param - lr * grad
        // For now, we'll implement a basic CPU-only version

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

        // Get mutable access to parameter data
        // For now, we'll implement a simplified version that doesn't require mutable access to Arc
        // In a full implementation, we would need to handle this differently
        Err(crate::error::MinitensorError::not_implemented(
            "Mutable tensor updates not yet implemented - requires tensor arithmetic operations",
        ))
    }

    /// Apply momentum-based SGD update
    fn apply_momentum_update(&mut self, param: &mut Tensor, grad: &Tensor, lr: f64) -> Result<()> {
        let param_id = param.id();

        // Get or create velocity buffer
        let _velocity = if let Some(v) = self.velocity.get(&param_id) {
            v.clone()
        } else {
            // Create new velocity buffer initialized to zeros
            let v = Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
            self.velocity.insert(param_id, v.clone());
            v
        };

        // For now, implement simple momentum update
        // In a full implementation: _velocity = momentum * _velocity + (1 - dampening) * grad
        // Then: param = param - lr * (grad + momentum * _velocity) for Nesterov
        // Or: param = param - lr * _velocity for standard momentum

        // Simplified implementation: just apply simple update for now
        self.apply_simple_update(param, grad, lr)
    }
}

impl Optimizer for SGD {
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
                // grad = grad + weight_decay * param
                self.apply_weight_decay(param, grad, weight_decay)?
            } else {
                grad.as_ref().clone()
            };

            // Apply momentum if configured
            if self.momentum > 0.0 {
                self.apply_momentum_update(param, &effective_grad, lr)?;
            } else {
                // Simple SGD update: param = param - lr * grad
                self.apply_simple_update(param, &effective_grad, lr)?;
            }
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
