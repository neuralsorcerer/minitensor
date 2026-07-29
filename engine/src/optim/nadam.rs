// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{GradientClipping, Optimizer, ParameterGroup};
use crate::{
    autograd::{self, TensorId},
    error::Result,
    ops::map::{PAR_CHUNK, PAR_THRESHOLD},
    tensor::Tensor,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// NAdam (Dozat, 2016): Adam with Nesterov momentum.
///
/// Plain Adam corrects the first moment by dividing out `1 - beta1^t`. Nesterov
/// momentum instead looks ahead — the step uses the momentum the *next* iterate
/// will carry rather than the current one, so it starts decelerating before it
/// overshoots rather than after.
///
/// The momentum coefficient is scheduled rather than fixed:
///
/// ```text
/// mu_t = beta1 * (1 - 0.5 * 0.96^(t * momentum_decay))
/// ```
///
/// which starts near `beta1 / 2` and rises toward `beta1`, damping the first few
/// steps while the moment estimates are still poor. The running product of every
/// `mu` so far replaces Adam's `beta1^t` in the bias correction, so it is kept as
/// optimizer state rather than recomputed.
pub struct NAdam {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index
    param_lookup: FxHashMap<TensorId, usize>,
    /// Default learning rate (for backward compatibility)
    default_lr: f64,
    /// Exponential decay rate for the first moment
    beta1: f64,
    /// Exponential decay rate for the second moment
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay coefficient
    weight_decay: f64,
    /// Controls how quickly the momentum schedule rises toward `beta1`
    momentum_decay: f64,
    /// First moment estimates
    m: FxHashMap<TensorId, Tensor>,
    /// Second moment estimates
    v: FxHashMap<TensorId, Tensor>,
    /// Running product of the momentum schedule, `prod_{i<=t} mu_i`
    mu_product: f64,
    /// Current step count
    step_count: usize,
    /// Gradient clipping configuration
    gradient_clipping: GradientClipping,
}

/// The scheduled momentum coefficients for a given step.
struct MomentumSchedule {
    /// `mu_t`
    current: f64,
    /// `mu_{t+1}`, the look-ahead coefficient Nesterov needs
    next: f64,
    /// `prod_{i<=t} mu_i`
    product: f64,
    /// `product * next`
    product_next: f64,
}

impl NAdam {
    /// Create a new NAdam optimizer with a single parameter group.
    pub fn new(
        learning_rate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
        momentum_decay: Option<f64>,
    ) -> Self {
        Self {
            param_groups: Vec::new(),
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
            momentum_decay: momentum_decay.unwrap_or(0.004),
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            mu_product: 1.0,
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

    /// Create a new NAdam optimizer with parameter groups
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, beta1: f64, beta2: f64) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.002);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 0.004,
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            mu_product: 1.0,
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        };
        optimizer.rebuild_param_lookup();
        optimizer
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    pub fn momentum_decay(&self) -> f64 {
        self.momentum_decay
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

    /// Advance the momentum schedule by one step and return this step's terms.
    ///
    /// The product accumulates across steps, so this must be called exactly once
    /// per `step` — never per parameter, or parameters later in the list would
    /// each see a further-advanced schedule.
    fn advance_schedule(&mut self) -> MomentumSchedule {
        let t = self.step_count as f64;
        let decay = |k: f64| self.beta1 * (1.0 - 0.5 * 0.96f64.powf(k * self.momentum_decay));
        let current = decay(t);
        let next = decay(t + 1.0);
        self.mu_product *= current;
        MomentumSchedule {
            current,
            next,
            product: self.mu_product,
            product_next: self.mu_product * next,
        }
    }

    /// Apply the NAdam update to one parameter
    fn apply_nadam_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
        schedule: &MomentumSchedule,
    ) -> Result<()> {
        let param_id = param.id();

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

        let m = self.m.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });
        let v = self.v.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        let beta2_correction = 1.0 - self.beta2.powi(self.step_count as i32);
        // The gradient and the momentum are scaled differently — that split is
        // what makes this Nesterov rather than plain Adam.
        let grad_scale = (1.0 - schedule.current) / (1.0 - schedule.product);
        let momentum_scale = schedule.next / (1.0 - schedule.product_next);

        macro_rules! nadam_arm {
            ($ty:ty, $read:ident, $write:ident) => {{
                let (lr, beta1, beta2, eps, wd): ($ty, $ty, $ty, $ty, $ty) = (
                    lr as $ty,
                    self.beta1 as $ty,
                    self.beta2 as $ty,
                    self.epsilon as $ty,
                    weight_decay as $ty,
                );
                let (grad_scale, momentum_scale, bc2): ($ty, $ty, $ty) = (
                    grad_scale as $ty,
                    momentum_scale as $ty,
                    beta2_correction as $ty,
                );
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let m_buf = m.data_mut().$write().unwrap();
                let v_buf = v.data_mut().$write().unwrap();
                let len = p.len();

                let step_chunk = |p: &mut [$ty], g: &[$ty], m: &mut [$ty], v: &mut [$ty]| {
                    for i in 0..p.len() {
                        let p_i = &mut p[i];
                        let g_val = g[i] + wd * *p_i;
                        m[i] = beta1 * m[i] + (1.0 - beta1) * g_val;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * g_val * g_val;
                        let denom = (v[i] / bc2).sqrt() + eps;
                        *p_i -= lr * (grad_scale * g_val + momentum_scale * m[i]) / denom;
                    }
                };

                // Below the threshold rayon's split overhead dwarfs the
                // arithmetic, so stay on the calling thread.
                if len < PAR_THRESHOLD {
                    step_chunk(p, g, m_buf, v_buf);
                } else {
                    p.par_chunks_mut(PAR_CHUNK)
                        .zip(g.par_chunks(PAR_CHUNK))
                        .zip(m_buf.par_chunks_mut(PAR_CHUNK))
                        .zip(v_buf.par_chunks_mut(PAR_CHUNK))
                        .for_each(|(((p, g), m), v)| step_chunk(p, g, m, v));
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => nadam_arm!(f32, as_f32_slice, as_f32_slice_mut),
            crate::tensor::DataType::Float64 => nadam_arm!(f64, as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "NAdam requires floating point parameters",
                ));
            }
        }

        Ok(())
    }
}

impl Optimizer for NAdam {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.clip_gradients(parameters, &self.gradient_clipping)?;

        self.step_count += 1;
        let schedule = self.advance_schedule();

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

            let lr = self.get_param_lr(param.id());
            let weight_decay = self.get_param_weight_decay(param.id());

            self.apply_nadam_update(param, &grad, lr, weight_decay, &schedule)?;
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
