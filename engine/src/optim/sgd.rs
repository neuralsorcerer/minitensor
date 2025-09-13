// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{GradientClipping, Optimizer, ParameterGroup};
use crate::{
    autograd::{self, TensorId},
    error::Result,
    tensor::Tensor,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// SGD optimizer with momentum support and parameter groups
pub struct SGD {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index
    param_lookup: FxHashMap<TensorId, usize>,
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
    velocity: FxHashMap<TensorId, Tensor>,
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
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            momentum: momentum.unwrap_or(0.0),
            weight_decay: weight_decay.unwrap_or(0.0),
            dampening: 0.0,
            nesterov: false,
            velocity: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Rebuild internal parameter lookup table
    fn rebuild_param_lookup(&mut self) {
        self.param_lookup.clear();
        let total: usize = self.param_groups.iter().map(|g| g.params.len()).sum();
        self.param_lookup.reserve(total);
        for (idx, group) in self.param_groups.iter().enumerate() {
            for &p in &group.params {
                self.param_lookup.insert(p, idx);
            }
        }
    }

    /// Create a new SGD optimizer with parameter groups
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, momentum: f64) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            momentum,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        };
        optimizer.rebuild_param_lookup();
        optimizer
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
        if let Some(&idx) = self.param_lookup.get(&param_id) {
            self.param_groups[idx].lr
        } else {
            self.default_lr
        }
    }

    /// Get weight decay for a specific parameter
    fn get_param_weight_decay(&self, param_id: TensorId) -> f64 {
        if let Some(&idx) = self.param_lookup.get(&param_id) {
            self.param_groups[idx].weight_decay
        } else {
            self.weight_decay
        }
    }

    /// Apply simple SGD update without momentum and optional weight decay
    fn apply_simple_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
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

        match param.dtype() {
            crate::tensor::DataType::Float32 => {
                let p = param.data_mut().as_f32_slice_mut().unwrap();
                let g = grad.data().as_f32_slice().unwrap();
                let lr = lr as f32;
                let wd = weight_decay as f32;
                p.par_iter_mut().zip(g.par_iter()).for_each(|(p_i, &g_i)| {
                    let grad_val = g_i + wd * *p_i;
                    *p_i -= lr * grad_val;
                });
            }
            crate::tensor::DataType::Float64 => {
                let p = param.data_mut().as_f64_slice_mut().unwrap();
                let g = grad.data().as_f64_slice().unwrap();
                p.par_iter_mut().zip(g.par_iter()).for_each(|(p_i, &g_i)| {
                    let grad_val = g_i + weight_decay * *p_i;
                    *p_i -= lr * grad_val;
                });
            }
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "SGD only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
    }

    /// Apply momentum-based SGD update
    fn apply_momentum_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let param_id = param.id();

        // Get or create velocity buffer
        let velocity = self.velocity.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        match param.dtype() {
            crate::tensor::DataType::Float32 => {
                let p = param.data_mut().as_f32_slice_mut().unwrap();
                let g = grad.data().as_f32_slice().unwrap();
                let v = velocity.data_mut().as_f32_slice_mut().unwrap();
                let lr = lr as f32;
                let momentum = self.momentum as f32;
                let damp = self.dampening as f32;
                let wd = weight_decay as f32;
                let nesterov = self.nesterov;
                p.par_iter_mut()
                    .zip(g.par_iter())
                    .zip(v.par_iter_mut())
                    .for_each(|((p_i, &g_i), v_i)| {
                        let grad_val = g_i + wd * *p_i;
                        *v_i = momentum * *v_i + (1.0 - damp) * grad_val;
                        let update = if nesterov {
                            grad_val + momentum * *v_i
                        } else {
                            *v_i
                        };
                        *p_i -= lr * update;
                    });
            }
            crate::tensor::DataType::Float64 => {
                let p = param.data_mut().as_f64_slice_mut().unwrap();
                let g = grad.data().as_f64_slice().unwrap();
                let v = velocity.data_mut().as_f64_slice_mut().unwrap();
                let momentum = self.momentum;
                let damp = self.dampening;
                let nesterov = self.nesterov;
                p.par_iter_mut()
                    .zip(g.par_iter())
                    .zip(v.par_iter_mut())
                    .for_each(|((p_i, &g_i), v_i)| {
                        let grad_val = g_i + weight_decay * *p_i;
                        *v_i = momentum * *v_i + (1.0 - damp) * grad_val;
                        let update = if nesterov {
                            grad_val + momentum * *v_i
                        } else {
                            *v_i
                        };
                        *p_i -= lr * update;
                    });
            }
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "SGD only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
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

            let grad = if let Some(g) = autograd::get_gradient(param) {
                g
            } else if let Some(g) = param.grad() {
                (**g).clone()
            } else {
                continue;
            };

            // Get learning rate for this parameter
            let lr = self.get_param_lr(param.id());
            let weight_decay = self.get_param_weight_decay(param.id());

            if self.momentum > 0.0 {
                self.apply_momentum_update(param, &grad, lr, weight_decay)?;
            } else {
                // Simple SGD update: param = param - lr * grad
                self.apply_simple_update(param, &grad, lr, weight_decay)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&self, parameters: &mut [&mut Tensor], set_to_none: bool) -> Result<()> {
        for param in parameters.iter_mut() {
            param.zero_grad(set_to_none);
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
        let idx = self.param_groups.len();
        for &p in &group.params {
            self.param_lookup.insert(p, idx);
        }
        self.param_groups.push(group);
        Ok(())
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}
