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

/// Adagrad optimizer with parameter groups.
///
/// The accumulator is a running *sum* of squared gradients rather than the
/// exponential moving average RMSprop keeps, so it never decreases. That is the
/// whole character of the method: each parameter's effective step
/// `lr / (sqrt(sum) + eps)` decays monotonically, quickly for parameters that
/// receive large or frequent gradients and barely at all for rarely-seen ones,
/// which is why it suits sparse features. It is also why Adagrad stalls on long
/// runs — the denominator grows without bound — and the reason the moving-average
/// methods exist.
pub struct Adagrad {
    /// Parameter groups with different learning rates
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index
    param_lookup: FxHashMap<TensorId, usize>,
    /// Default learning rate (for backward compatibility)
    default_lr: f64,
    /// Decay applied to the learning rate as steps accumulate
    lr_decay: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Weight decay coefficient
    weight_decay: f64,
    /// Value the accumulator starts at
    initial_accumulator_value: f64,
    /// Running sum of squared gradients, per parameter
    state_sum: FxHashMap<TensorId, Tensor>,
    /// Current step count
    step_count: usize,
    /// Gradient clipping configuration
    gradient_clipping: GradientClipping,
}

impl Adagrad {
    /// Create a new Adagrad optimizer with a single parameter group.
    ///
    /// `epsilon` defaults to 1e-10 rather than the 1e-8 the moving-average
    /// optimizers use: Adagrad's denominator is a sum that only grows, so a
    /// larger floor would keep biting long after it stopped being needed.
    pub fn new(
        learning_rate: f64,
        lr_decay: Option<f64>,
        weight_decay: Option<f64>,
        initial_accumulator_value: Option<f64>,
        epsilon: Option<f64>,
    ) -> Self {
        Self {
            param_groups: Vec::new(),
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            lr_decay: lr_decay.unwrap_or(0.0),
            epsilon: epsilon.unwrap_or(1e-10),
            weight_decay: weight_decay.unwrap_or(0.0),
            initial_accumulator_value: initial_accumulator_value.unwrap_or(0.0),
            state_sum: FxHashMap::default(),
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

    /// Create a new Adagrad optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        lr_decay: f64,
        epsilon: f64,
    ) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(0.01);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            lr_decay,
            epsilon,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            state_sum: FxHashMap::default(),
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

    /// Get the learning-rate decay coefficient
    pub fn lr_decay(&self) -> f64 {
        self.lr_decay
    }

    /// Set the learning-rate decay coefficient
    pub fn set_lr_decay(&mut self, lr_decay: f64) {
        self.lr_decay = lr_decay;
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

    /// Get the value new accumulators start at
    pub fn initial_accumulator_value(&self) -> f64 {
        self.initial_accumulator_value
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

    /// Apply the Adagrad update to one parameter
    fn apply_adagrad_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
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

        let initial = self.initial_accumulator_value;
        let state_sum = self.state_sum.entry(param_id).or_insert_with(|| {
            let zeros = Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
            if initial == 0.0 {
                zeros
            } else {
                fill_with(zeros, initial)
            }
        });

        // The step count is already incremented for this step, so `step_count`
        // is 1 on the first update and the decay factor starts at exactly `lr`.
        let decayed_lr = lr / (1.0 + (self.step_count.saturating_sub(1) as f64) * self.lr_decay);

        macro_rules! adagrad_arm {
            ($ty:ty, $read:ident, $write:ident, $lr:expr, $eps:expr, $wd:expr) => {{
                let (lr, eps, wd): ($ty, $ty, $ty) = ($lr, $eps, $wd);
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let sum = state_sum.data_mut().$write().unwrap();
                let len = p.len();

                let step_chunk = |p: &mut [$ty], g: &[$ty], sum: &mut [$ty]| {
                    for i in 0..p.len() {
                        let p_i = &mut p[i];
                        let g_val = g[i] + wd * *p_i;
                        sum[i] += g_val * g_val;
                        *p_i -= lr * g_val / (sum[i].sqrt() + eps);
                    }
                };

                // Below the threshold rayon's split overhead dwarfs the
                // arithmetic, so stay on the calling thread.
                if len < PAR_THRESHOLD {
                    step_chunk(p, g, sum);
                } else {
                    p.par_chunks_mut(PAR_CHUNK)
                        .zip(g.par_chunks(PAR_CHUNK))
                        .zip(sum.par_chunks_mut(PAR_CHUNK))
                        .for_each(|((p, g), sum)| step_chunk(p, g, sum));
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => adagrad_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                decayed_lr as f32,
                self.epsilon as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => adagrad_arm!(
                f64,
                as_f64_slice,
                as_f64_slice_mut,
                decayed_lr,
                self.epsilon,
                weight_decay
            ),
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "Adagrad requires floating point parameters",
                ));
            }
        }

        Ok(())
    }
}

/// Set every element of an (owned, freshly allocated) tensor to `value`.
fn fill_with(mut tensor: Tensor, value: f64) -> Tensor {
    match tensor.dtype() {
        crate::tensor::DataType::Float32 => {
            if let Some(slice) = tensor.data_mut().as_f32_slice_mut() {
                slice.fill(value as f32);
            }
        }
        crate::tensor::DataType::Float64 => {
            if let Some(slice) = tensor.data_mut().as_f64_slice_mut() {
                slice.fill(value);
            }
        }
        _ => {}
    }
    tensor
}

impl Optimizer for Adagrad {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.clip_gradients(parameters, &self.gradient_clipping)?;

        self.step_count += 1;

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

            self.apply_adagrad_update(param, &grad, lr, weight_decay)?;
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
