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

/// RMSprop optimizer with parameter groups
pub struct RMSprop {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index
    param_lookup: FxHashMap<TensorId, usize>,
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
    square_avg: FxHashMap<TensorId, Tensor>,
    /// Momentum buffers
    momentum_buffer: FxHashMap<TensorId, Tensor>,
    /// Gradient average buffers (for centered variant)
    grad_avg: FxHashMap<TensorId, Tensor>,
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
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            alpha: alpha.unwrap_or(0.99),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            momentum: momentum.unwrap_or(0.0),
            centered: false,
            square_avg: FxHashMap::default(),
            momentum_buffer: FxHashMap::default(),
            grad_avg: FxHashMap::default(),
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

    /// Create a new RMSprop optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        alpha: f64,
        epsilon: f64,
        momentum: f64,
    ) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            alpha,
            epsilon,
            weight_decay: 0.0,
            momentum,
            centered: false,
            square_avg: FxHashMap::default(),
            momentum_buffer: FxHashMap::default(),
            grad_avg: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        };
        optimizer.rebuild_param_lookup();
        optimizer
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

    /// Apply RMSprop optimization update
    fn apply_rmsprop_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let param_id = param.id();

        // Get or create square average buffer
        let square_avg = self.square_avg.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        // Get or create momentum buffer if momentum > 0
        let momentum_buffer_opt = if self.momentum > 0.0 {
            Some(self.momentum_buffer.entry(param_id).or_insert_with(|| {
                Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
            }))
        } else {
            None
        };

        // Get or create gradient average buffer for centered variant
        let grad_avg_opt = if self.centered {
            Some(self.grad_avg.entry(param_id).or_insert_with(|| {
                Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
            }))
        } else {
            None
        };

        // Perform RMSprop update directly
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

        let alpha = self.alpha;
        let eps = self.epsilon;

        match param.dtype() {
            crate::tensor::DataType::Float32 => {
                let p = param.data_mut().as_f32_slice_mut().unwrap();
                let g = grad.data().as_f32_slice().unwrap();
                let sq = square_avg.data_mut().as_f32_slice_mut().unwrap();
                let lr = lr as f32;
                let momentum = self.momentum as f32;
                let wd = weight_decay as f32;
                match (momentum_buffer_opt, grad_avg_opt) {
                    (Some(mb), Some(ga)) => {
                        let mb = mb.data_mut().as_f32_slice_mut().unwrap();
                        let ga = ga.data_mut().as_f32_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(mb.par_iter_mut())
                            .zip(ga.par_iter_mut())
                            .for_each(|((((p_i, &g_i), sq_i), mb_i), ga_i)| {
                                let g_val = g_i + wd * *p_i;
                                *sq_i = alpha as f32 * *sq_i + (1.0 - alpha as f32) * g_val * g_val;
                                *ga_i = alpha as f32 * *ga_i + (1.0 - alpha as f32) * g_val;
                                let avg = *sq_i - *ga_i * *ga_i;
                                let denom = avg.sqrt() + eps as f32;
                                *mb_i = momentum * *mb_i + lr * g_val / denom;
                                *p_i -= *mb_i;
                            });
                    }
                    (Some(mb), None) => {
                        let mb = mb.data_mut().as_f32_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(mb.par_iter_mut())
                            .for_each(|(((p_i, &g_i), sq_i), mb_i)| {
                                let g_val = g_i + wd * *p_i;
                                *sq_i = alpha as f32 * *sq_i + (1.0 - alpha as f32) * g_val * g_val;
                                let denom = sq_i.sqrt() + eps as f32;
                                *mb_i = momentum * *mb_i + lr * g_val / denom;
                                *p_i -= *mb_i;
                            });
                    }
                    (None, Some(ga)) => {
                        let ga = ga.data_mut().as_f32_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(ga.par_iter_mut())
                            .for_each(|(((p_i, &g_i), sq_i), ga_i)| {
                                let g_val = g_i + wd * *p_i;
                                *sq_i = alpha as f32 * *sq_i + (1.0 - alpha as f32) * g_val * g_val;
                                *ga_i = alpha as f32 * *ga_i + (1.0 - alpha as f32) * g_val;
                                let avg = *sq_i - *ga_i * *ga_i;
                                let denom = avg.sqrt() + eps as f32;
                                *p_i -= lr * g_val / denom;
                            });
                    }
                    (None, None) => {
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .for_each(|((p_i, &g_i), sq_i)| {
                                let g_val = g_i + wd * *p_i;
                                *sq_i = alpha as f32 * *sq_i + (1.0 - alpha as f32) * g_val * g_val;
                                let denom = sq_i.sqrt() + eps as f32;
                                *p_i -= lr * g_val / denom;
                            });
                    }
                }
            }
            crate::tensor::DataType::Float64 => {
                let p = param.data_mut().as_f64_slice_mut().unwrap();
                let g = grad.data().as_f64_slice().unwrap();
                let sq = square_avg.data_mut().as_f64_slice_mut().unwrap();
                let lr = lr;
                let momentum = self.momentum;
                match (momentum_buffer_opt, grad_avg_opt) {
                    (Some(mb), Some(ga)) => {
                        let mb = mb.data_mut().as_f64_slice_mut().unwrap();
                        let ga = ga.data_mut().as_f64_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(mb.par_iter_mut())
                            .zip(ga.par_iter_mut())
                            .for_each(|((((p_i, &g_i), sq_i), mb_i), ga_i)| {
                                let g_val = g_i + weight_decay * *p_i;
                                *sq_i = alpha * *sq_i + (1.0 - alpha) * g_val * g_val;
                                *ga_i = alpha * *ga_i + (1.0 - alpha) * g_val;
                                let avg = *sq_i - *ga_i * *ga_i;
                                let denom = avg.sqrt() + eps;
                                *mb_i = momentum * *mb_i + lr * g_val / denom;
                                *p_i -= *mb_i;
                            });
                    }
                    (Some(mb), None) => {
                        let mb = mb.data_mut().as_f64_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(mb.par_iter_mut())
                            .for_each(|(((p_i, &g_i), sq_i), mb_i)| {
                                let g_val = g_i + weight_decay * *p_i;
                                *sq_i = alpha * *sq_i + (1.0 - alpha) * g_val * g_val;
                                let denom = sq_i.sqrt() + eps;
                                *mb_i = momentum * *mb_i + lr * g_val / denom;
                                *p_i -= *mb_i;
                            });
                    }
                    (None, Some(ga)) => {
                        let ga = ga.data_mut().as_f64_slice_mut().unwrap();
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .zip(ga.par_iter_mut())
                            .for_each(|(((p_i, &g_i), sq_i), ga_i)| {
                                let g_val = g_i + weight_decay * *p_i;
                                *sq_i = alpha * *sq_i + (1.0 - alpha) * g_val * g_val;
                                *ga_i = alpha * *ga_i + (1.0 - alpha) * g_val;
                                let avg = *sq_i - *ga_i * *ga_i;
                                let denom = avg.sqrt() + eps;
                                *p_i -= lr * g_val / denom;
                            });
                    }
                    (None, None) => {
                        p.par_iter_mut()
                            .zip(g.par_iter())
                            .zip(sq.par_iter_mut())
                            .for_each(|((p_i, &g_i), sq_i)| {
                                let g_val = g_i + weight_decay * *p_i;
                                *sq_i = alpha * *sq_i + (1.0 - alpha) * g_val * g_val;
                                let denom = sq_i.sqrt() + eps;
                                *p_i -= lr * g_val / denom;
                            });
                    }
                }
            }
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "RMSprop only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
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

            // Apply RMSprop update
            self.apply_rmsprop_update(param, &grad, lr, weight_decay)?;
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
