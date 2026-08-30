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
use std::collections::hash_map::Entry;

/// SGD optimizer with momentum support and parameter groups
pub struct SGD {
    /// Parameter groups, their reverse index, and the defaults for a
    /// parameter in none of them.
    groups: ParamGroups,
    /// Momentum coefficient
    momentum: f64,
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
        let mut groups = ParamGroups::new(learning_rate);
        groups.set_default_weight_decay(weight_decay.unwrap_or(0.0));
        Self {
            groups,
            momentum: momentum.unwrap_or(0.0),
            dampening: 0.0,
            nesterov: false,
            velocity: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new SGD optimizer with parameter groups
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, momentum: f64) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.001),
            momentum,
            dampening: 0.0,
            nesterov: false,
            velocity: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
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
        self.groups.default_weight_decay()
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.groups.set_default_weight_decay(weight_decay);
    }

    /// Get the momentum dampening coefficient
    pub fn dampening(&self) -> f64 {
        self.dampening
    }

    /// Check if using Nesterov momentum
    pub fn is_nesterov(&self) -> bool {
        self.nesterov
    }

    /// Apply simple SGD update without momentum and optional weight decay
    fn apply_simple_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        check_param_grad_match(param, grad)?;

        /// One dtype arm. The math lives in a single chunk closure; the chunk
        /// loop stays on the calling thread for small parameters (biases, norm
        /// scales, small layers), where rayon's split overhead dwarfs the
        /// arithmetic, and fans out only above `PAR_THRESHOLD`.
        macro_rules! simple_arm {
            ($ty:ty, $read:ident, $write:ident, $lr:expr, $wd:expr) => {{
                let (lr, wd): ($ty, $ty) = ($lr, $wd);
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let step_chunk = |p: &mut [$ty], g: &[$ty]| {
                    for (p_i, &g_i) in p.iter_mut().zip(g.iter()) {
                        let grad_val = g_i + wd * *p_i;
                        *p_i -= lr * grad_val;
                    }
                };
                if p.len() < PAR_THRESHOLD {
                    step_chunk(p, g);
                } else {
                    par_param_update(p, g, &mut [], PAR_CHUNK, &|p, g, _| step_chunk(p, g));
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => simple_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                lr as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => {
                simple_arm!(f64, as_f64_slice, as_f64_slice_mut, lr, weight_decay)
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
        check_param_grad_match(param, grad)?;

        let param_id = param.id();

        // Get or create velocity buffer. `is_new` marks a freshly created (or
        // reset) buffer: on that first step the momentum buffer is the
        // gradient itself (`buf = grad.clone()`), applying the
        // `(1 - dampening)` factor only from the second step onward. Tracking
        // this keeps the first step undamped.
        let (velocity, is_new) = match self.velocity.entry(param_id) {
            Entry::Occupied(mut entry) => {
                let needs_reset = entry.get().shape() != param.shape()
                    || entry.get().dtype() != param.dtype()
                    || entry.get().device() != param.device();
                if needs_reset {
                    entry.insert(Tensor::zeros(
                        param.shape().clone(),
                        param.dtype(),
                        param.device(),
                        false,
                    ));
                }
                (entry.into_mut(), needs_reset)
            }
            Entry::Vacant(entry) => (
                entry.insert(Tensor::zeros(
                    param.shape().clone(),
                    param.dtype(),
                    param.device(),
                    false,
                )),
                true,
            ),
        };

        let nesterov = self.nesterov;

        /// One dtype arm; see `simple_arm!` above for the threshold rationale.
        macro_rules! momentum_arm {
            ($ty:ty, $read:ident, $write:ident, $lr:expr, $mom:expr, $damp:expr, $wd:expr) => {{
                let (lr, momentum, damp, wd): ($ty, $ty, $ty, $ty) = ($lr, $mom, $damp, $wd);
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let v = velocity.data_mut().$write().unwrap();
                let step_chunk = |p: &mut [$ty], g: &[$ty], v: &mut [$ty]| {
                    for ((p_i, &g_i), v_i) in p.iter_mut().zip(g.iter()).zip(v.iter_mut()) {
                        let grad_val = g_i + wd * *p_i;
                        *v_i = if is_new {
                            grad_val
                        } else {
                            momentum * *v_i + (1.0 - damp) * grad_val
                        };
                        let update = if nesterov {
                            grad_val + momentum * *v_i
                        } else {
                            *v_i
                        };
                        *p_i -= lr * update;
                    }
                };
                if p.len() < PAR_THRESHOLD {
                    step_chunk(p, g, v);
                } else {
                    par_param_update(p, g, &mut [v], PAR_CHUNK, &|p, g, state| {
                        let [v] = state else {
                            unreachable!("one state buffer")
                        };
                        step_chunk(p, g, v)
                    });
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => momentum_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                lr as f32,
                self.momentum as f32,
                self.dampening as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => momentum_arm!(
                f64,
                as_f64_slice,
                as_f64_slice_mut,
                lr,
                self.momentum,
                self.dampening,
                weight_decay
            ),
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
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("SGD", self.step_count, parameters.len());
        save_param_buffers(&mut state, "momentum_buffer", &self.velocity, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("SGD", parameters.len())?;
        load_param_buffers(state, "momentum_buffer", &mut self.velocity, parameters)?;
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

            if self.momentum > 0.0 {
                self.apply_momentum_update(param, &grad, lr, weight_decay)?;
            } else {
                // Simple SGD update: param = param - lr * grad
                self.apply_simple_update(param, &grad, lr, weight_decay)?;
            }
        }

        Ok(())
    }

    fn describe(&self) -> String {
        format!(
            "SGD(lr={:?}, momentum={:?}, dampening={:?}, weight_decay={:?}, nesterov={})",
            self.learning_rate(),
            self.momentum(),
            self.dampening(),
            self.weight_decay(),
            super::optimizer::py_bool(self.is_nesterov())
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
