// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::{
    GradientClipping, Optimizer, ParamGroups, ParameterGroup, check_param_grad_match,
    parameter_gradient,
};
use super::utils::{load_param_buffers, save_param_buffers};
use crate::serialization::OptimizerState;
use crate::{
    autograd::TensorId,
    error::Result,
    ops::map::{PAR_CHUNK, PAR_THRESHOLD, par_param_update},
    tensor::Tensor,
};
use rustc_hash::FxHashMap;

/// Adam optimizer with bias correction and parameter groups
pub struct Adam {
    /// Parameter groups, their reverse index, and the defaults for a
    /// parameter in none of them.
    groups: ParamGroups,
    /// Beta1 coefficient for first moment estimates
    beta1: f64,
    /// Beta2 coefficient for second moment estimates
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Whether to use AMSGrad variant
    amsgrad: bool,
    /// Whether to use decoupled weight decay (AdamW)
    decoupled_weight_decay: bool,
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
            groups: {
                let mut g = ParamGroups::new(learning_rate);
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            amsgrad: false,
            decoupled_weight_decay: false,
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            v_hat: FxHashMap::default(),
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
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.001),
            beta1,
            beta2,
            epsilon,
            amsgrad: false,
            decoupled_weight_decay: false,
            m: FxHashMap::default(),
            v: FxHashMap::default(),
            v_hat: FxHashMap::default(),
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

    /// Which algorithm a checkpoint written by this optimizer belongs to.
    ///
    /// AdamW shares Adam's buffer layout but not its update rule -- decoupled
    /// weight decay is applied to the parameter rather than folded into the
    /// gradient -- so resuming one from the other's checkpoint would silently
    /// continue a different optimisation.
    fn algorithm_name(&self) -> &'static str {
        if self.decoupled_weight_decay {
            "AdamW"
        } else {
            "Adam"
        }
    }

    /// Enable or disable decoupled weight decay (AdamW)
    pub fn with_decoupled_weight_decay(mut self, enabled: bool) -> Self {
        self.decoupled_weight_decay = enabled;
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
        self.groups.default_weight_decay()
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.groups.set_default_weight_decay(weight_decay);
    }

    /// Check if using AMSGrad
    pub fn is_amsgrad(&self) -> bool {
        self.amsgrad
    }

    /// Check if decoupled weight decay is enabled
    pub fn is_decoupled_weight_decay(&self) -> bool {
        self.decoupled_weight_decay
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
        check_param_grad_match(param, grad)?;

        let step = self.step_count as i32;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.epsilon;
        let beta1_pow = beta1.powi(step);
        let beta2_pow = beta2.powi(step);
        let bc1_inv = 1.0 / (1.0 - beta1_pow);
        let bc2_inv = 1.0 / (1.0 - beta2_pow);
        let use_decoupled_weight_decay = self.decoupled_weight_decay && weight_decay != 0.0;

        /// One dtype arm. The per-element math lives in a single closure over
        /// whole chunks; the chunk loop runs on the calling thread for small
        /// parameters — the common case (biases, norm scales, small layers),
        /// where rayon's split overhead dwarfed the arithmetic — and fans out
        /// over rayon only once the tensor is large enough to pay for it.
        macro_rules! adam_arm {
            ($ty:ty, $read:ident, $write:ident, $lr:expr, $b1:expr, $b2:expr, $bc1:expr,
             $bc2:expr, $eps:expr, $wd:expr) => {{
                let (lr, beta1, beta2, bc1_inv, bc2_inv, eps, wd): (
                    $ty,
                    $ty,
                    $ty,
                    $ty,
                    $ty,
                    $ty,
                    $ty,
                ) = ($lr, $b1, $b2, $bc1, $bc2, $eps, $wd);
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let m_buf = m.data_mut().$write().unwrap();
                let v_buf = v.data_mut().$write().unwrap();
                let len = p.len();

                let step_chunk = |p: &mut [$ty],
                                  g: &[$ty],
                                  m: &mut [$ty],
                                  v: &mut [$ty],
                                  mut vhat: Option<&mut [$ty]>| {
                    for i in 0..p.len() {
                        let p_i = &mut p[i];
                        if use_decoupled_weight_decay {
                            *p_i -= lr * wd * *p_i;
                        }
                        let g_val = if use_decoupled_weight_decay {
                            g[i]
                        } else {
                            g[i] + wd * *p_i
                        };
                        m[i] = beta1 * m[i] + (1.0 - beta1) * g_val;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * g_val * g_val;
                        let second_moment = match vhat.as_deref_mut() {
                            Some(vhat) => {
                                if v[i] > vhat[i] {
                                    vhat[i] = v[i];
                                }
                                vhat[i]
                            }
                            None => v[i],
                        };
                        let m_hat = m[i] * bc1_inv;
                        let v_hat_corr = second_moment * bc2_inv;
                        *p_i -= lr * m_hat / (v_hat_corr.sqrt() + eps);
                    }
                };

                match v_hat_opt {
                    Some(vhat) => {
                        let vhat = vhat.data_mut().$write().unwrap();
                        if len < PAR_THRESHOLD {
                            step_chunk(p, g, m_buf, v_buf, Some(vhat));
                        } else {
                            let state = &mut [m_buf, v_buf, vhat];
                            par_param_update(p, g, state, PAR_CHUNK, &|p, g, state| {
                                let [m, v, vhat] = state else {
                                    unreachable!("three state buffers")
                                };
                                step_chunk(p, g, m, v, Some(vhat))
                            });
                        }
                    }
                    None => {
                        if len < PAR_THRESHOLD {
                            step_chunk(p, g, m_buf, v_buf, None);
                        } else {
                            par_param_update(
                                p,
                                g,
                                &mut [m_buf, v_buf],
                                PAR_CHUNK,
                                &|p, g, state| {
                                    let [m, v] = state else {
                                        unreachable!("two state buffers")
                                    };
                                    step_chunk(p, g, m, v, None)
                                },
                            );
                        }
                    }
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => adam_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                lr as f32,
                beta1 as f32,
                beta2 as f32,
                bc1_inv as f32,
                bc2_inv as f32,
                eps as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => adam_arm!(
                f64,
                as_f64_slice,
                as_f64_slice_mut,
                lr,
                beta1,
                beta2,
                bc1_inv,
                bc2_inv,
                eps,
                weight_decay
            ),
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "Adam only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
    }
}

/// Decoupled Adam optimizer (AdamW)
pub struct AdamW {
    inner: Adam,
}

impl AdamW {
    /// Create a new AdamW optimizer with single parameter group
    pub fn new(
        learning_rate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        let adam = Adam::new(learning_rate, beta1, beta2, epsilon, weight_decay)
            .with_decoupled_weight_decay(true);
        Self { inner: adam }
    }

    /// Create a new AdamW optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let adam = Adam::with_param_groups(param_groups, beta1, beta2, epsilon)
            .with_decoupled_weight_decay(true);
        Self { inner: adam }
    }

    /// Get beta1 coefficient
    pub fn beta1(&self) -> f64 {
        self.inner.beta1()
    }

    /// Get beta2 coefficient
    pub fn beta2(&self) -> f64 {
        self.inner.beta2()
    }

    /// Get epsilon value
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    /// Get weight decay coefficient
    pub fn weight_decay(&self) -> f64 {
        self.inner.weight_decay()
    }

    /// Get the learning rate (for single parameter group optimizers)
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    /// Set the learning rate (for single parameter group optimizers)
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr);
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.inner.step(parameters)
    }

    fn zero_grad(&self, parameters: &mut [&mut Tensor], set_to_none: bool) -> Result<()> {
        self.inner.zero_grad(parameters, set_to_none)
    }

    fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr)
    }

    fn param_groups(&self) -> &[ParameterGroup] {
        self.inner.param_groups()
    }

    fn param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        self.inner.param_groups_mut()
    }

    fn add_param_group(&mut self, group: ParameterGroup) -> Result<()> {
        self.inner.add_param_group(group)
    }

    fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        self.inner.state_dict(parameters)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        self.inner.load_state_dict(parameters, state)
    }

    fn clip_gradients(
        &self,
        parameters: &mut [&mut Tensor],
        clipping: &GradientClipping,
    ) -> Result<()> {
        self.inner.clip_gradients(parameters, clipping)
    }
}

impl Optimizer for Adam {
    /// Buffer names are the conventional ones (`exp_avg`, `exp_avg_sq`,
    /// `max_exp_avg_sq`) so a checkpoint is readable by anyone who has seen
    /// one before.
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state =
            OptimizerState::new(self.algorithm_name(), self.step_count, parameters.len());
        save_param_buffers(&mut state, "exp_avg", &self.m, parameters)?;
        save_param_buffers(&mut state, "exp_avg_sq", &self.v, parameters)?;
        save_param_buffers(&mut state, "max_exp_avg_sq", &self.v_hat, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible(self.algorithm_name(), parameters.len())?;
        load_param_buffers(state, "exp_avg", &mut self.m, parameters)?;
        load_param_buffers(state, "exp_avg_sq", &mut self.v, parameters)?;
        load_param_buffers(state, "max_exp_avg_sq", &mut self.v_hat, parameters)?;
        self.step_count = state.step_count;
        Ok(())
    }

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

            let Some(grad) = parameter_gradient(param) else {
                continue;
            };

            // Get learning rate for this parameter
            let lr = self.groups.lr(param.id());
            let weight_decay = self.groups.weight_decay(param.id());

            // Apply Adam update with weight decay
            self.apply_adam_update(param, &grad, lr, weight_decay)?;
        }

        Ok(())
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
