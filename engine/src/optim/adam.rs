// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{GradientClipping, Optimizer, ParameterGroup};
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use std::collections::HashMap;

/// Adam optimizer with bias correction and parameter groups
pub struct Adam {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Default learning rate (for backward compatibility)
    default_lr: f64,
    /// Beta1 coefficient for first moment estimates
    beta1: f64,
    /// Beta2 coefficient for second moment estimates
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay coefficient
    weight_decay: f64,
    /// Whether to use AMSGrad variant
    amsgrad: bool,
    /// First moment estimates
    m: HashMap<TensorId, Tensor>,
    /// Second moment estimates
    v: HashMap<TensorId, Tensor>,
    /// Maximum second moment estimates (for AMSGrad)
    v_hat: HashMap<TensorId, Tensor>,
    /// Current step count
    step_count: usize,
    /// Gradient clipping configuration
    gradient_clipping: GradientClipping,
}

impl Adam {
    /// Create a new Adam optimizer with single parameter group
    pub fn new(
        learning_rate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            param_groups: Vec::new(),
            default_lr: learning_rate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            amsgrad: false,
            m: HashMap::new(),
            v: HashMap::new(),
            v_hat: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Adam optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        Self {
            param_groups,
            default_lr,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
            m: HashMap::new(),
            v: HashMap::new(),
            v_hat: HashMap::new(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Enable AMSGrad variant
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get beta1 coefficient
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Set beta1 coefficient
    pub fn set_beta1(&mut self, beta1: f64) {
        self.beta1 = beta1;
    }

    /// Get beta2 coefficient
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Set beta2 coefficient
    pub fn set_beta2(&mut self, beta2: f64) {
        self.beta2 = beta2;
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

    /// Check if using AMSGrad
    pub fn is_amsgrad(&self) -> bool {
        self.amsgrad
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

    /// Apply Adam optimization update
    fn apply_adam_update(&mut self, param: &mut Tensor, grad: &Tensor, lr: f64) -> Result<()> {
        let param_id = param.id();

        // Get or create first moment estimate (m)
        let m = if let Some(moment) = self.m.get(&param_id) {
            moment.clone()
        } else {
            let moment = Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
            self.m.insert(param_id, moment.clone());
            moment
        };

        // Get or create second moment estimate (v)
        let v = if let Some(moment) = self.v.get(&param_id) {
            moment.clone()
        } else {
            let moment = Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
            self.v.insert(param_id, moment.clone());
            moment
        };

        // Get or create max second moment estimate (v_hat) for AMSGrad
        let v_hat = if self.amsgrad {
            if let Some(moment) = self.v_hat.get(&param_id) {
                moment.clone()
            } else {
                let moment =
                    Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
                self.v_hat.insert(param_id, moment.clone());
                moment
            }
        } else {
            // Dummy tensor for non-AMSGrad case
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        };

        // Perform Adam update
        self.adam_step(param, grad, &m, &v, &v_hat, lr)?;

        Ok(())
    }

    /// Perform the actual Adam optimization step
    fn adam_step(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        _m: &Tensor,
        _v: &Tensor,
        _v_hat: &Tensor,
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

impl Optimizer for Adam {
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

            // Apply Adam update
            self.apply_adam_update(param, &effective_grad, lr)?;
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
