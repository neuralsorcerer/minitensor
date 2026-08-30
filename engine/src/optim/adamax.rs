// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{
    GradientClipping, Optimizer, ParamGroups, ParameterGroup, check_param_grad_match,
};
use super::utils::{
    fixed_state_update, load_param_buffers, save_param_buffers, step_each_parameter,
};
use crate::serialization::OptimizerState;
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use rustc_hash::FxHashMap;

/// Adamax: Adam with the second moment measured by the infinity norm.
///
/// Adam divides by a decaying root-mean-square of past gradients. Adamax
/// divides by a decaying *maximum* instead -- `u = max(beta2 * u, |g|)` -- so
/// one enormous gradient sets the denominator and then decays out of it
/// geometrically, rather than being squared into an average that takes far
/// longer to forget. There is also no second bias correction to apply: a
/// maximum of a decaying sequence is not shrunk towards zero by starting at
/// zero the way a mean is.
pub struct Adamax {
    groups: ParamGroups,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    /// Exponential moving average of gradients, per parameter
    exp_avg: FxHashMap<TensorId, Tensor>,
    /// Decaying maximum of gradient magnitudes, per parameter
    exp_inf: FxHashMap<TensorId, Tensor>,
    step_count: usize,
    gradient_clipping: GradientClipping,
}

impl Adamax {
    /// Create a new Adamax optimizer with a single parameter group.
    pub fn new(
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            groups: {
                let mut g = ParamGroups::new(learning_rate.unwrap_or(0.002));
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            exp_avg: FxHashMap::default(),
            exp_inf: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Adamax optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.002),
            beta1,
            beta2,
            epsilon,
            exp_avg: FxHashMap::default(),
            exp_inf: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get the first-moment decay rate
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Get the infinity-norm decay rate
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get weight decay coefficient
    pub fn weight_decay(&self) -> f64 {
        self.groups.default_weight_decay()
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.groups.set_default_weight_decay(weight_decay);
    }

    fn apply_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        check_param_grad_match(param, grad)?;
        let param_id = param.id();
        let zeros = || Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
        let exp_avg = self.exp_avg.entry(param_id).or_insert_with(zeros);
        let exp_inf = self.exp_inf.entry(param_id).or_insert_with(zeros);

        // The one bias correction Adamax needs, folded into the step size so
        // the element loop does no work that is the same for every element.
        let bias_correction = 1.0 - self.beta1.powi(self.step_count as i32);
        let corrected_lr = lr / bias_correction;

        fixed_state_update!(
            "Adamax",
            param,
            grad,
            [exp_avg, exp_inf],
            [
                lr = corrected_lr,
                beta1 = self.beta1,
                beta2 = self.beta2,
                eps = self.epsilon,
                wd = weight_decay
            ],
            |p, g, i, [avg, inf]| {
                let value = &mut p[i];
                let gradient = g[i] + wd * *value;
                avg[i] = beta1 * avg[i] + (1.0 - beta1) * gradient;
                // The infinity norm: whichever is larger, the decayed running
                // maximum or this gradient's magnitude. `max` here also keeps
                // a NaN gradient from silently becoming the denominator, since
                // a comparison against NaN is false and the running value
                // wins.
                let decayed = beta2 * inf[i];
                let magnitude = gradient.abs();
                inf[i] = if magnitude > decayed {
                    magnitude
                } else {
                    decayed
                };
                *value -= lr * avg[i] / (inf[i] + eps);
            }
        )
    }
}

impl Optimizer for Adamax {
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("Adamax", self.step_count, parameters.len());
        save_param_buffers(&mut state, "exp_avg", &self.exp_avg, parameters)?;
        save_param_buffers(&mut state, "exp_inf", &self.exp_inf, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("Adamax", parameters.len())?;
        load_param_buffers(state, "exp_avg", &mut self.exp_avg, parameters)?;
        load_param_buffers(state, "exp_inf", &mut self.exp_inf, parameters)?;
        self.step_count = state.step_count;
        Ok(())
    }

    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.clip_gradients(parameters, &self.gradient_clipping)?;
        self.step_count += 1;

        step_each_parameter(parameters, |param, grad| {
            let lr = self.groups.lr(param.id());
            let weight_decay = self.groups.weight_decay(param.id());
            self.apply_update(param, grad, lr, weight_decay)
        })
    }

    fn describe(&self) -> String {
        format!(
            "Adamax(lr={:?}, betas=({}, {}), eps={:?}, weight_decay={:?})",
            self.learning_rate(),
            self.beta1(),
            self.beta2(),
            self.epsilon(),
            self.weight_decay()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
