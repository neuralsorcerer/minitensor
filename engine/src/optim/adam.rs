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

/// Adam optimizer with bias correction and parameter groups
pub struct Adam {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index
    param_lookup: FxHashMap<TensorId, usize>,
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
    m: FxHashMap<TensorId, Tensor>,
    /// Second moment estimates
    v: FxHashMap<TensorId, Tensor>,
    /// Maximum second moment estimates (for AMSGrad)
    v_hat: FxHashMap<TensorId, Tensor>,
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
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            amsgrad: false,
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            v_hat: FxHashMap::default(),
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

    /// Create a new Adam optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.001);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            v_hat: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        };
        optimizer.rebuild_param_lookup();
        optimizer
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

    /// Apply Adam optimization update
    fn apply_adam_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let param_id = param.id();

        // Get or create first moment estimate (m)
        let m = self.m.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        // Get or create second moment estimate (v)
        let v = self.v.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        // Get or create max second moment estimate (v_hat) for AMSGrad
        let v_hat_opt = if self.amsgrad {
            Some(self.v_hat.entry(param_id).or_insert_with(|| {
                Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
            }))
        } else {
            None
        };

        // Perform Adam update directly
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

        let step = self.step_count as i32;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.epsilon;
        let beta1_pow = beta1.powi(step);
        let beta2_pow = beta2.powi(step);
        let bc1_inv = 1.0 / (1.0 - beta1_pow);
        let bc2_inv = 1.0 / (1.0 - beta2_pow);

        match param.dtype() {
            crate::tensor::DataType::Float32 => {
                let p = param.data_mut().as_f32_slice_mut().unwrap();
                let g = grad.data().as_f32_slice().unwrap();
                let m_buf = m.data_mut().as_f32_slice_mut().unwrap();
                let v_buf = v.data_mut().as_f32_slice_mut().unwrap();
                let lr = lr as f32;
                let beta1_f = beta1 as f32;
                let beta2_f = beta2 as f32;
                let wd = weight_decay as f32;
                let bc1_inv = bc1_inv as f32;
                let bc2_inv = bc2_inv as f32;
                if let Some(vhat) = v_hat_opt {
                    let vhat_slice = vhat.data_mut().as_f32_slice_mut().unwrap();
                    p.par_iter_mut()
                        .zip(g.par_iter())
                        .zip(m_buf.par_iter_mut())
                        .zip(v_buf.par_iter_mut())
                        .zip(vhat_slice.par_iter_mut())
                        .for_each(|((((p_i, &g_i), m_i), v_i), vhat_i)| {
                            let g_val = g_i + wd * *p_i;
                            *m_i = beta1_f * *m_i + (1.0 - beta1_f) * g_val;
                            *v_i = beta2_f * *v_i + (1.0 - beta2_f) * g_val * g_val;
                            if *v_i > *vhat_i {
                                *vhat_i = *v_i;
                            }
                            let m_hat = *m_i * bc1_inv;
                            let v_hat_corr = *vhat_i * bc2_inv;
                            *p_i -= lr * m_hat / (v_hat_corr.sqrt() + eps as f32);
                        });
                } else {
                    p.par_iter_mut()
                        .zip(g.par_iter())
                        .zip(m_buf.par_iter_mut())
                        .zip(v_buf.par_iter_mut())
                        .for_each(|(((p_i, &g_i), m_i), v_i)| {
                            let g_val = g_i + wd * *p_i;
                            *m_i = beta1_f * *m_i + (1.0 - beta1_f) * g_val;
                            *v_i = beta2_f * *v_i + (1.0 - beta2_f) * g_val * g_val;
                            let m_hat = *m_i * bc1_inv;
                            let v_hat_corr = *v_i * bc2_inv;
                            *p_i -= lr * m_hat / (v_hat_corr.sqrt() + eps as f32);
                        });
                }
            }
            crate::tensor::DataType::Float64 => {
                let p = param.data_mut().as_f64_slice_mut().unwrap();
                let g = grad.data().as_f64_slice().unwrap();
                let m_buf = m.data_mut().as_f64_slice_mut().unwrap();
                let v_buf = v.data_mut().as_f64_slice_mut().unwrap();
                if let Some(vhat) = v_hat_opt {
                    let vhat_slice = vhat.data_mut().as_f64_slice_mut().unwrap();
                    p.par_iter_mut()
                        .zip(g.par_iter())
                        .zip(m_buf.par_iter_mut())
                        .zip(v_buf.par_iter_mut())
                        .zip(vhat_slice.par_iter_mut())
                        .for_each(|((((p_i, &g_i), m_i), v_i), vhat_i)| {
                            let g_val = g_i + weight_decay * *p_i;
                            *m_i = beta1 * *m_i + (1.0 - beta1) * g_val;
                            *v_i = beta2 * *v_i + (1.0 - beta2) * g_val * g_val;
                            if *v_i > *vhat_i {
                                *vhat_i = *v_i;
                            }
                            let m_hat = *m_i * bc1_inv;
                            let v_hat_corr = *vhat_i * bc2_inv;
                            *p_i -= lr * m_hat / (v_hat_corr.sqrt() + eps);
                        });
                } else {
                    p.par_iter_mut()
                        .zip(g.par_iter())
                        .zip(m_buf.par_iter_mut())
                        .zip(v_buf.par_iter_mut())
                        .for_each(|(((p_i, &g_i), m_i), v_i)| {
                            let g_val = g_i + weight_decay * *p_i;
                            *m_i = beta1 * *m_i + (1.0 - beta1) * g_val;
                            *v_i = beta2 * *v_i + (1.0 - beta2) * g_val * g_val;
                            let m_hat = *m_i * bc1_inv;
                            let v_hat_corr = *v_i * bc2_inv;
                            *p_i -= lr * m_hat / (v_hat_corr.sqrt() + eps);
                        });
                }
            }
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "Adam only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
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

            // Apply Adam update with weight decay
            self.apply_adam_update(param, &grad, lr, weight_decay)?;
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
