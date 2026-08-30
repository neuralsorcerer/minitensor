// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{
    GradientClipping, Optimizer, ParamGroups, ParameterGroup, check_param_grad_match,
    parameter_gradient,
};
use super::utils::{fixed_state_update, load_param_buffers, save_param_buffers};
use crate::serialization::OptimizerState;
use crate::{autograd::TensorId, error::Result, tensor::Tensor};
use rustc_hash::FxHashMap;

/// RAdam: Adam with its early steps corrected for a variance nobody has
/// measured yet.
///
/// Adam's adaptive step divides by an estimate of the gradient's second
/// moment. In the first few steps that estimate is built from almost no
/// samples, so its own variance is enormous and the resulting steps are wild
/// -- which is what a linear warmup schedule exists to paper over. RAdam
/// computes how many samples the estimate effectively has and scales the step
/// by the correction that variance implies, so the warmup falls out of the
/// method instead of being tuned into it. Below five effective samples there
/// is no usable estimate at all and it takes a plain, non-adaptive step.
pub struct RAdam {
    groups: ParamGroups,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    /// Exponential moving average of gradients, per parameter
    exp_avg: FxHashMap<TensorId, Tensor>,
    /// Exponential moving average of squared gradients, per parameter
    exp_avg_sq: FxHashMap<TensorId, Tensor>,
    step_count: usize,
    gradient_clipping: GradientClipping,
}

/// The step size and whether the adaptive denominator is usable at step `t`.
struct Rectification {
    /// Multiplier on the bias-corrected first moment.
    scale: f64,
    /// `None` once the variance estimate has too few effective samples to
    /// divide by, in which case the step is the plain moving average.
    second_moment_correction: Option<f64>,
}

impl RAdam {
    /// Create a new RAdam optimizer with a single parameter group.
    pub fn new(
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            groups: {
                let mut g = ParamGroups::new(learning_rate.unwrap_or(0.001));
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            exp_avg: FxHashMap::default(),
            exp_avg_sq: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new RAdam optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.001),
            beta1,
            beta2,
            epsilon,
            exp_avg: FxHashMap::default(),
            exp_avg_sq: FxHashMap::default(),
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

    /// Get the second-moment decay rate
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

    /// How far to move at step `t`, and whether to divide by the variance
    /// estimate at all.
    ///
    /// Everything here depends on the step number and nothing on the
    /// parameter, so it is computed once per step rather than once per
    /// element.
    fn rectification(&self, lr: f64) -> Rectification {
        let step = self.step_count as f64;
        let bias_correction1 = 1.0 - self.beta1.powf(step);
        let beta2_t = self.beta2.powf(step);
        let bias_correction2 = 1.0 - beta2_t;

        // The number of samples the second-moment estimate effectively has:
        // its maximum, less the shortfall the geometric weighting leaves at
        // this step.
        let rho_infinity = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho = rho_infinity - 2.0 * step * beta2_t / bias_correction2;

        // Below five the variance of the estimate is not defined -- the term
        // under the root goes negative -- so the paper takes the plain
        // non-adaptive step there, which is what makes the first few steps a
        // warmup without one being scheduled.
        if rho <= 5.0 {
            return Rectification {
                scale: lr / bias_correction1,
                second_moment_correction: None,
            };
        }

        let rectifier = (((rho - 4.0) * (rho - 2.0) * rho_infinity)
            / ((rho_infinity - 4.0) * (rho_infinity - 2.0) * rho))
            .sqrt();
        Rectification {
            scale: lr * rectifier / bias_correction1,
            second_moment_correction: Some(bias_correction2.sqrt()),
        }
    }

    fn apply_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        check_param_grad_match(param, grad)?;
        let rectification = self.rectification(lr);
        let param_id = param.id();
        let zeros = || Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false);
        let exp_avg = self.exp_avg.entry(param_id).or_insert_with(zeros);
        let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(zeros);

        // Zero says "take the plain step": multiplying the denominator's
        // square root by it and adding one leaves a divisor of exactly 1, so
        // the two branches are one line of arithmetic rather than a branch
        // inside the element loop.
        let (adaptive, root_correction) = match rectification.second_moment_correction {
            Some(correction) => (1.0, correction),
            None => (0.0, 1.0),
        };

        fixed_state_update!(
            "RAdam",
            param,
            grad,
            [exp_avg, exp_avg_sq],
            [
                lr = rectification.scale,
                beta1 = self.beta1,
                beta2 = self.beta2,
                eps = self.epsilon,
                wd = weight_decay,
                adaptive = adaptive,
                root_correction = root_correction
            ],
            |p, g, i, [avg, avg_sq]| {
                let value = &mut p[i];
                let gradient = g[i] + wd * *value;
                avg[i] = beta1 * avg[i] + (1.0 - beta1) * gradient;
                avg_sq[i] = beta2 * avg_sq[i] + (1.0 - beta2) * gradient * gradient;
                let denominator =
                    1.0 - adaptive + adaptive * (avg_sq[i].sqrt() / root_correction + eps);
                *value -= lr * avg[i] / denominator;
            }
        )
    }
}

impl Optimizer for RAdam {
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("RAdam", self.step_count, parameters.len());
        save_param_buffers(&mut state, "exp_avg", &self.exp_avg, parameters)?;
        save_param_buffers(&mut state, "exp_avg_sq", &self.exp_avg_sq, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("RAdam", parameters.len())?;
        load_param_buffers(state, "exp_avg", &mut self.exp_avg, parameters)?;
        load_param_buffers(state, "exp_avg_sq", &mut self.exp_avg_sq, parameters)?;
        self.step_count = state.step_count;
        Ok(())
    }

    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.clip_gradients(parameters, &self.gradient_clipping)?;
        self.step_count += 1;

        for param in parameters.iter_mut() {
            if !param.requires_grad() {
                continue;
            }
            let Some(grad) = parameter_gradient(param) else {
                continue;
            };
            let lr = self.groups.lr(param.id());
            let weight_decay = self.groups.weight_decay(param.id());
            self.apply_update(param, &grad, lr, weight_decay)?;
        }

        Ok(())
    }

    fn describe(&self) -> String {
        format!(
            "RAdam(lr={:?}, betas=({}, {}), eps={:?}, weight_decay={:?})",
            self.learning_rate(),
            self.beta1(),
            self.beta2(),
            self.epsilon(),
            self.weight_decay()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
