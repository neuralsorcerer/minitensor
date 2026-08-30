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
use smallvec::SmallVec;

/// RMSprop optimizer with parameter groups
pub struct RMSprop {
    /// Parameter groups, their reverse index, and the defaults for a
    /// parameter in none of them.
    groups: ParamGroups,
    /// Alpha coefficient for moving average
    alpha: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
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
            groups: {
                let mut g = ParamGroups::new(learning_rate);
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            alpha: alpha.unwrap_or(0.99),
            epsilon: epsilon.unwrap_or(1e-8),
            momentum: momentum.unwrap_or(0.0),
            centered: false,
            square_avg: FxHashMap::default(),
            momentum_buffer: FxHashMap::default(),
            grad_avg: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new RMSprop optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        alpha: f64,
        epsilon: f64,
        momentum: f64,
    ) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.001),
            alpha,
            epsilon,
            momentum,
            centered: false,
            square_avg: FxHashMap::default(),
            momentum_buffer: FxHashMap::default(),
            grad_avg: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
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
        self.groups.default_weight_decay()
    }

    /// Set weight decay coefficient
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.groups.set_default_weight_decay(weight_decay);
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
        check_param_grad_match(param, grad)?;

        /// One dtype arm. The four state combinations (momentum buffer and/or
        /// centering average) used to be four near-identical rayon pipelines
        /// per dtype — eight copies of the same recurrence. They collapse into
        /// a single chunk closure that branches on the optional buffers, and
        /// the chunk loop stays on the calling thread for small parameters,
        /// where rayon's split overhead dwarfs the arithmetic.
        macro_rules! rmsprop_arm {
            ($ty:ty, $read:ident, $write:ident, $lr:expr, $alpha:expr, $mom:expr, $eps:expr,
             $wd:expr) => {{
                let (lr, alpha, momentum, eps, wd): ($ty, $ty, $ty, $ty, $ty) =
                    ($lr, $alpha, $mom, $eps, $wd);
                let one_minus_alpha = 1.0 - alpha;
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let sq = square_avg.data_mut().$write().unwrap();
                let mut mb = momentum_buffer_opt.map(|t| t.data_mut().$write().unwrap());
                let mut ga = grad_avg_opt.map(|t| t.data_mut().$write().unwrap());
                let len = p.len();

                let step_chunk = |p: &mut [$ty],
                                  g: &[$ty],
                                  sq: &mut [$ty],
                                  mut mb: Option<&mut [$ty]>,
                                  mut ga: Option<&mut [$ty]>| {
                    for i in 0..p.len() {
                        let p_i = &mut p[i];
                        let g_val = g[i] + wd * *p_i;
                        sq[i] = alpha * sq[i] + one_minus_alpha * g_val * g_val;
                        // Centered RMSprop divides by the running *variance*:
                        // mean of squares minus the square of the mean.
                        let variance = match ga.as_deref_mut() {
                            Some(ga) => {
                                ga[i] = alpha * ga[i] + one_minus_alpha * g_val;
                                sq[i] - ga[i] * ga[i]
                            }
                            None => sq[i],
                        };
                        let denom = variance.sqrt() + eps;
                        match mb.as_deref_mut() {
                            Some(mb) => {
                                mb[i] = momentum * mb[i] + g_val / denom;
                                *p_i -= lr * mb[i];
                            }
                            None => *p_i -= lr * g_val / denom,
                        }
                    }
                };

                if len < PAR_THRESHOLD {
                    step_chunk(p, g, sq, mb.as_deref_mut(), ga.as_deref_mut());
                } else {
                    // Only the buffers actually in use are handed over, so the
                    // arity of `state` names the configuration. This used to
                    // build a `Vec<Option<&mut [T]>>` of one entry per chunk
                    // per optional buffer -- thousands of allocations per step
                    // on a large parameter -- purely so a single zipped
                    // pipeline could cover all four combinations.
                    let centered = ga.is_some();
                    let mut state: SmallVec<[&mut [$ty]; 3]> = SmallVec::new();
                    state.push(sq);
                    if let Some(mb) = mb.as_deref_mut() {
                        state.push(mb);
                    }
                    if let Some(ga) = ga.as_deref_mut() {
                        state.push(ga);
                    }
                    par_param_update(p, g, &mut state, PAR_CHUNK, &|p, g, state| match state {
                        [sq] => step_chunk(p, g, sq, None, None),
                        [sq, ga] if centered => step_chunk(p, g, sq, None, Some(ga)),
                        [sq, mb] => step_chunk(p, g, sq, Some(mb), None),
                        [sq, mb, ga] => step_chunk(p, g, sq, Some(mb), Some(ga)),
                        _ => unreachable!("rmsprop passes one to three state buffers"),
                    });
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => rmsprop_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                lr as f32,
                self.alpha as f32,
                self.momentum as f32,
                self.epsilon as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => rmsprop_arm!(
                f64,
                as_f64_slice,
                as_f64_slice_mut,
                lr,
                self.alpha,
                self.momentum,
                self.epsilon,
                weight_decay
            ),
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
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("RMSprop", self.step_count, parameters.len());
        save_param_buffers(&mut state, "square_avg", &self.square_avg, parameters)?;
        save_param_buffers(
            &mut state,
            "momentum_buffer",
            &self.momentum_buffer,
            parameters,
        )?;
        save_param_buffers(&mut state, "grad_avg", &self.grad_avg, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("RMSprop", parameters.len())?;
        load_param_buffers(state, "square_avg", &mut self.square_avg, parameters)?;
        load_param_buffers(
            state,
            "momentum_buffer",
            &mut self.momentum_buffer,
            parameters,
        )?;
        load_param_buffers(state, "grad_avg", &mut self.grad_avg, parameters)?;
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

            // Apply RMSprop update
            self.apply_rmsprop_update(param, &grad, lr, weight_decay)?;
        }

        Ok(())
    }

    fn describe(&self) -> String {
        format!(
            "RMSprop(lr={:?}, alpha={:?}, eps={:?}, weight_decay={:?}, momentum={:?}, centered={})",
            self.learning_rate(),
            self.alpha(),
            self.epsilon(),
            self.weight_decay(),
            self.momentum(),
            super::optimizer::py_bool(self.is_centered())
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
