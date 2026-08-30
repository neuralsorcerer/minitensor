// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{
    GradientClipping, Optimizer, ParamGroups, ParameterGroup, check_param_grad_match,
};
use super::utils::{load_param_buffers, save_param_buffers, step_each_parameter};
use crate::serialization::OptimizerState;
use crate::{
    autograd::TensorId,
    error::Result,
    ops::map::{PAR_CHUNK, PAR_THRESHOLD, par_param_update},
    tensor::Tensor,
};
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
    /// Parameter groups, their reverse index, and the defaults for a
    /// parameter in none of them.
    groups: ParamGroups,
    /// Decay applied to the learning rate as steps accumulate
    lr_decay: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
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
            groups: {
                let mut g = ParamGroups::new(learning_rate);
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            lr_decay: lr_decay.unwrap_or(0.0),
            epsilon: epsilon.unwrap_or(1e-10),
            initial_accumulator_value: initial_accumulator_value.unwrap_or(0.0),
            state_sum: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Adagrad optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        lr_decay: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.01),
            lr_decay,
            epsilon,
            initial_accumulator_value: 0.0,
            state_sum: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
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
        self.groups.default_weight_decay()
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.groups.set_default_weight_decay(weight_decay);
    }

    /// Get the value new accumulators start at
    pub fn initial_accumulator_value(&self) -> f64 {
        self.initial_accumulator_value
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

        check_param_grad_match(param, grad)?;

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
                    par_param_update(p, g, &mut [sum], PAR_CHUNK, &|p, g, state| {
                        let [sum] = state else {
                            unreachable!("one state buffer")
                        };
                        step_chunk(p, g, sum)
                    });
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
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("Adagrad", self.step_count, parameters.len());
        save_param_buffers(&mut state, "sum", &self.state_sum, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("Adagrad", parameters.len())?;
        load_param_buffers(state, "sum", &mut self.state_sum, parameters)?;
        self.step_count = state.step_count;
        Ok(())
    }

    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.clip_gradients(parameters, &self.gradient_clipping)?;
        self.step_count += 1;

        step_each_parameter(parameters, |param, grad| {
            let lr = self.groups.lr(param.id());
            let weight_decay = self.groups.weight_decay(param.id());
            self.apply_adagrad_update(param, grad, lr, weight_decay)
        })
    }

    fn describe(&self) -> String {
        format!(
            "Adagrad(lr={:?}, lr_decay={:?}, eps={:?})",
            self.learning_rate(),
            self.lr_decay(),
            self.epsilon()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
