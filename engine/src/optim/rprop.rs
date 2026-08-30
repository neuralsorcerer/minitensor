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

/// Rprop: resilient backpropagation, which uses only the *sign* of the
/// gradient.
///
/// Every other optimizer here scales its step by the gradient's magnitude, so
/// a flat region gives tiny steps and a cliff gives enormous ones. Rprop keeps
/// a step size per parameter and moves by exactly that, in the gradient's
/// direction: the magnitude never enters. A step whose direction agrees with
/// the last one grows by `eta_plus`; one that reverses shrinks by `eta_minus`
/// and is not taken at all, because reversing means the last step went past a
/// minimum and repeating it in reverse would go past it again.
///
/// That makes it immune to badly scaled gradients and useless on mini-batches:
/// the sign of a noisy gradient flips for reasons that have nothing to do with
/// the surface, so the step sizes collapse. It is a full-batch method.
pub struct Rprop {
    groups: ParamGroups,
    /// Growth factor for a step that agreed with the last one
    eta_plus: f64,
    /// Shrink factor for a step that reversed
    eta_minus: f64,
    /// Smallest step size a parameter can shrink to
    step_min: f64,
    /// Largest step size a parameter can grow to
    step_max: f64,
    /// The previous step's gradient, whose sign decides what happens to the
    /// step size
    prev_grad: FxHashMap<TensorId, Tensor>,
    /// The per-parameter step size, which is the whole state of the method
    step_size: FxHashMap<TensorId, Tensor>,
    step_count: usize,
    gradient_clipping: GradientClipping,
}

impl Rprop {
    /// Create a new Rprop optimizer with a single parameter group.
    ///
    /// `learning_rate` is the size every step starts at, not a scale on the
    /// gradient -- there is no gradient magnitude in this method to scale.
    pub fn new(
        learning_rate: Option<f64>,
        eta_minus: Option<f64>,
        eta_plus: Option<f64>,
        step_min: Option<f64>,
        step_max: Option<f64>,
    ) -> Self {
        Self {
            groups: ParamGroups::new(learning_rate.unwrap_or(0.01)),
            eta_plus: eta_plus.unwrap_or(1.2),
            eta_minus: eta_minus.unwrap_or(0.5),
            step_min: step_min.unwrap_or(1e-6),
            step_max: step_max.unwrap_or(50.0),
            prev_grad: FxHashMap::default(),
            step_size: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Rprop optimizer with parameter groups
    pub fn with_param_groups(
        param_groups: Vec<ParameterGroup>,
        eta_minus: f64,
        eta_plus: f64,
        step_min: f64,
        step_max: f64,
    ) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 0.01),
            eta_plus,
            eta_minus,
            step_min,
            step_max,
            prev_grad: FxHashMap::default(),
            step_size: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Set gradient clipping
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// Get the shrink factor for a reversed step
    pub fn eta_minus(&self) -> f64 {
        self.eta_minus
    }

    /// Get the growth factor for an agreeing step
    pub fn eta_plus(&self) -> f64 {
        self.eta_plus
    }

    /// Get the smallest allowed step size
    pub fn step_min(&self) -> f64 {
        self.step_min
    }

    /// Get the largest allowed step size
    pub fn step_max(&self) -> f64 {
        self.step_max
    }

    fn apply_update(&mut self, param: &mut Tensor, grad: &Tensor, lr: f64) -> Result<()> {
        check_param_grad_match(param, grad)?;
        let param_id = param.id();
        let shape = param.shape().clone();
        let dtype = param.dtype();
        let device = param.device();
        let prev_grad = self
            .prev_grad
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(shape.clone(), dtype, device, false));
        // Every step starts at `lr`, so the buffer is filled rather than
        // zeroed: a step size of zero would never move and never grow.
        let step_size = self.step_size.entry(param_id).or_insert_with(|| {
            let mut buffer = Tensor::zeros(shape, dtype, device, false);
            match dtype {
                crate::tensor::DataType::Float32 => {
                    if let Some(slice) = buffer.data_mut().as_f32_slice_mut() {
                        slice.fill(lr as f32);
                    }
                }
                crate::tensor::DataType::Float64 => {
                    if let Some(slice) = buffer.data_mut().as_f64_slice_mut() {
                        slice.fill(lr);
                    }
                }
                _ => {}
            }
            buffer
        });

        fixed_state_update!(
            "Rprop",
            param,
            grad,
            [prev_grad, step_size],
            [
                eta_plus = self.eta_plus,
                eta_minus = self.eta_minus,
                step_min = self.step_min,
                step_max = self.step_max
            ],
            |p, g, i, [previous, size]| {
                let agreement = g[i] * previous[i];
                // A gradient that kept its direction earns a longer step; one
                // that reversed means the last step overshot, so the step is
                // shortened and *not taken* -- and the gradient is forgotten,
                // so the step after it cannot count the same reversal twice.
                let gradient = if agreement > 0.0 {
                    size[i] = (size[i] * eta_plus).min(step_max);
                    g[i]
                } else if agreement < 0.0 {
                    size[i] = (size[i] * eta_minus).max(step_min);
                    0.0
                } else {
                    // Exactly zero: either the first step, or a gradient that
                    // has reached zero. The step size stands.
                    g[i]
                };
                // Only the sign moves the parameter. `signum` would answer 1
                // for a zero gradient and 1 for a NaN, so the comparison is
                // written out: a zero gradient must not move anything.
                if gradient > 0.0 {
                    p[i] -= size[i];
                } else if gradient < 0.0 {
                    p[i] += size[i];
                }
                previous[i] = gradient;
            }
        )
    }
}

impl Optimizer for Rprop {
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("Rprop", self.step_count, parameters.len());
        save_param_buffers(&mut state, "prev", &self.prev_grad, parameters)?;
        save_param_buffers(&mut state, "step_size", &self.step_size, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("Rprop", parameters.len())?;
        load_param_buffers(state, "prev", &mut self.prev_grad, parameters)?;
        load_param_buffers(state, "step_size", &mut self.step_size, parameters)?;
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
            self.apply_update(param, &grad, lr)?;
        }

        Ok(())
    }

    fn describe(&self) -> String {
        format!(
            "Rprop(lr={:?}, etas=({}, {}), step_sizes=({:?}, {:?}))",
            self.learning_rate(),
            self.eta_minus(),
            self.eta_plus(),
            self.step_min(),
            self.step_max()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}
