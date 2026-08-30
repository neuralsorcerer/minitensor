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

/// Adadelta, which needs no learning rate to be tuned.
///
/// Adagrad's step shrinks without bound because its denominator is a sum that
/// only grows; RMSprop fixes that with a decaying average but leaves the step
/// dimensionally wrong -- a gradient divided by a gradient is a pure number,
/// so the step is measured in nothing at all and the learning rate has to
/// carry the units. Adadelta divides by the same running gradient magnitude
/// and multiplies by a running magnitude of its own past *steps*, so the
/// answer is measured in the units of the parameter and `lr` defaults to 1.
pub struct Adadelta {
    groups: ParamGroups,
    /// Decay for both running averages
    rho: f64,
    /// Added under both square roots, so it also seeds the very first step
    epsilon: f64,
    /// Running average of squared gradients, per parameter
    square_avg: FxHashMap<TensorId, Tensor>,
    /// Running average of squared *updates*, which is what supplies the units
    acc_delta: FxHashMap<TensorId, Tensor>,
    step_count: usize,
    gradient_clipping: GradientClipping,
}

impl Adadelta {
    /// Create a new Adadelta optimizer with a single parameter group.
    ///
    /// `learning_rate` defaults to 1: the method already produces a step in
    /// the parameter's own units, so this is a plain multiplier on it rather
    /// than the scale that decides whether the run converges.
    pub fn new(
        learning_rate: Option<f64>,
        rho: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            groups: {
                let mut g = ParamGroups::new(learning_rate.unwrap_or(1.0));
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            rho: rho.unwrap_or(0.9),
            epsilon: epsilon.unwrap_or(1e-6),
            square_avg: FxHashMap::default(),
            acc_delta: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Adadelta optimizer with parameter groups
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, rho: f64, epsilon: f64) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 1.0),
            rho,
            epsilon,
            square_avg: FxHashMap::default(),
            acc_delta: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get the decay for both running averages
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Set the decay for both running averages
    pub fn set_rho(&mut self, rho: f64) {
        self.rho = rho;
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
        // Two maps rather than one of pairs, so each buffer is borrowed on its
        // own and `save_param_buffers` can name them separately in a
        // checkpoint.
        let square_avg = self.square_avg.entry(param_id).or_insert_with(zeros);
        let acc_delta = self.acc_delta.entry(param_id).or_insert_with(zeros);

        fixed_state_update!(
            "Adadelta",
            param,
            grad,
            [square_avg, acc_delta],
            [
                lr = lr,
                rho = self.rho,
                eps = self.epsilon,
                wd = weight_decay
            ],
            |p, g, i, [square, delta]| {
                let value = &mut p[i];
                let gradient = g[i] + wd * *value;
                square[i] = rho * square[i] + (1.0 - rho) * gradient * gradient;
                // The ratio of two root-mean-squares: the numerator carries
                // the units of a parameter and the denominator those of a
                // gradient, which is what leaves the step in the parameter's
                // own units.
                let step = ((delta[i] + eps).sqrt() / (square[i] + eps).sqrt()) * gradient;
                delta[i] = rho * delta[i] + (1.0 - rho) * step * step;
                *value -= lr * step;
            }
        )
    }
}

impl Optimizer for Adadelta {
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("Adadelta", self.step_count, parameters.len());
        save_param_buffers(&mut state, "square_avg", &self.square_avg, parameters)?;
        save_param_buffers(&mut state, "acc_delta", &self.acc_delta, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("Adadelta", parameters.len())?;
        load_param_buffers(state, "square_avg", &mut self.square_avg, parameters)?;
        load_param_buffers(state, "acc_delta", &mut self.acc_delta, parameters)?;
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
            "Adadelta(lr={:?}, rho={:?}, eps={:?}, weight_decay={:?})",
            self.learning_rate(),
            self.rho(),
            self.epsilon(),
            self.weight_decay()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
