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

/// Lion optimizer (EvoLved Sign Momentum; Chen et al., "Symbolic Discovery of
/// Optimization Algorithms", 2023).
///
/// Lion was found by symbolic program search and is both simpler and more
/// memory-efficient than Adam: it keeps a single momentum buffer per parameter
/// (half the state of Adam) and its update is the *sign* of an interpolated
/// momentum, so every parameter moves by a uniform `lr` magnitude. It matches or
/// beats AdamW on large-scale vision and language models.
///
/// Per parameter `θ` with gradient `g`:
/// ```text
/// c   = β1·m + (1 - β1)·g          # interpolated update direction
/// θ   = θ - lr·(sign(c) + λ·θ)     # decoupled weight decay λ
/// m   = β2·m + (1 - β2)·g          # momentum uses a slower β2
/// ```
/// Because the update is sign-based, the effective step size is `lr`; a Lion
/// learning rate is therefore typically ~3-10× smaller than the AdamW one, with
/// a correspondingly larger weight decay.
pub struct Lion {
    /// Parameter groups, their reverse index, and the defaults for a
    /// parameter in none of them.
    groups: ParamGroups,
    /// Momentum interpolation coefficient for the update direction.
    beta1: f64,
    /// Momentum EMA coefficient for the stored buffer.
    beta2: f64,
    /// Momentum buffers, one per parameter.
    m: FxHashMap<TensorId, Tensor>,
    /// Number of steps taken.
    step_count: usize,
    /// Gradient clipping configuration.
    gradient_clipping: GradientClipping,
}

#[inline(always)]
fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn sign_f64(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl Lion {
    /// Create a new Lion optimizer with a single parameter group.
    pub fn new(
        learning_rate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            groups: {
                let mut g = ParamGroups::new(learning_rate);
                g.set_default_weight_decay(weight_decay.unwrap_or(0.0));
                g
            },
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.99),
            m: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Create a new Lion optimizer with explicit parameter groups.
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, beta1: f64, beta2: f64) -> Self {
        Self {
            groups: ParamGroups::from_groups(param_groups, 1e-4),
            beta1,
            beta2,
            m: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Configure gradient clipping.
    pub fn with_gradient_clipping(mut self, clipping: GradientClipping) -> Self {
        self.gradient_clipping = clipping;
        self
    }

    /// First momentum coefficient.
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Second momentum coefficient.
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Weight decay coefficient.
    pub fn weight_decay(&self) -> f64 {
        self.groups.default_weight_decay()
    }

    /// Apply the Lion update to a single parameter in place.
    fn apply_lion_update(
        &mut self,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f64,
        weight_decay: f64,
    ) -> Result<()> {
        let param_id = param.id();
        let m = self.m.entry(param_id).or_insert_with(|| {
            Tensor::zeros(param.shape().clone(), param.dtype(), param.device(), false)
        });

        check_param_grad_match(param, grad)?;

        let beta1 = self.beta1;
        let beta2 = self.beta2;

        /// One dtype arm. The math lives in a single chunk closure; the chunk
        /// loop stays on the calling thread for small parameters, where
        /// rayon's split overhead dwarfs the arithmetic, and fans out only
        /// above `PAR_THRESHOLD`.
        macro_rules! lion_arm {
            ($ty:ty, $read:ident, $write:ident, $sign:ident, $lr:expr, $b1:expr, $b2:expr,
             $wd:expr) => {{
                let (lr, beta1, beta2, wd): ($ty, $ty, $ty, $ty) = ($lr, $b1, $b2, $wd);
                let p = param.data_mut().$write().unwrap();
                let g = grad.data().$read().unwrap();
                let m_buf = m.data_mut().$write().unwrap();
                let step_chunk = |p: &mut [$ty], g: &[$ty], m: &mut [$ty]| {
                    for ((p_i, &g_i), m_i) in p.iter_mut().zip(g.iter()).zip(m.iter_mut()) {
                        let m_old = *m_i;
                        let update = $sign(beta1 * m_old + (1.0 - beta1) * g_i);
                        *p_i -= lr * (update + wd * *p_i);
                        *m_i = beta2 * m_old + (1.0 - beta2) * g_i;
                    }
                };
                if p.len() < PAR_THRESHOLD {
                    step_chunk(p, g, m_buf);
                } else {
                    par_param_update(p, g, &mut [m_buf], PAR_CHUNK, &|p, g, state| {
                        let [m] = state else {
                            unreachable!("one state buffer")
                        };
                        step_chunk(p, g, m)
                    });
                }
            }};
        }

        match param.dtype() {
            crate::tensor::DataType::Float32 => lion_arm!(
                f32,
                as_f32_slice,
                as_f32_slice_mut,
                sign_f32,
                lr as f32,
                beta1 as f32,
                beta2 as f32,
                weight_decay as f32
            ),
            crate::tensor::DataType::Float64 => lion_arm!(
                f64,
                as_f64_slice,
                as_f64_slice_mut,
                sign_f64,
                lr,
                beta1,
                beta2,
                weight_decay
            ),
            _ => {
                return Err(crate::error::MinitensorError::invalid_operation(
                    "Lion only supports float32/float64 tensors",
                ));
            }
        }

        Ok(())
    }
}

impl Optimizer for Lion {
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let mut state = OptimizerState::new("Lion", self.step_count, parameters.len());
        save_param_buffers(&mut state, "exp_avg", &self.m, parameters)?;
        Ok(state)
    }

    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        state.check_compatible("Lion", parameters.len())?;
        load_param_buffers(state, "exp_avg", &mut self.m, parameters)?;
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
            self.apply_lion_update(param, &grad, lr, weight_decay)?;
        }

        Ok(())
    }

    fn describe(&self) -> String {
        format!(
            "Lion(lr={:?}, betas=({}, {}), weight_decay={:?})",
            self.learning_rate(),
            self.beta1(),
            self.beta2(),
            self.weight_decay()
        )
    }

    crate::delegate_optimizer_bookkeeping!(groups, step_count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::{DataType, Shape};

    fn param(data: Vec<f64>, grad: Vec<f64>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let mut t = Tensor::zeros(shape.clone(), DataType::Float64, Device::cpu(), true);
        t.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        let mut g = Tensor::zeros(shape, DataType::Float64, Device::cpu(), false);
        g.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&grad);
        t.set_grad(Some(g));
        t
    }

    #[test]
    fn first_step_is_signed_lr() {
        // With m0 = 0, c = (1-β1)·g, sign(c) = sign(g). No weight decay, so each
        // param moves by exactly -lr·sign(g).
        let mut t = param(vec![1.0, 2.0, -3.0], vec![0.5, -0.1, 4.0]);
        let mut opt = Lion::new(0.1, None, None, None);
        {
            let mut refs: Vec<&mut Tensor> = vec![&mut t];
            opt.step(refs.as_mut_slice()).unwrap();
        }
        let got = t.data().as_f64_slice().unwrap().to_vec();
        assert!((got[0] - (1.0 - 0.1)).abs() < 1e-12);
        assert!((got[1] - (2.0 + 0.1)).abs() < 1e-12);
        assert!((got[2] - (-3.0 - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn zero_gradient_leaves_params_unchanged() {
        let mut t = param(vec![1.0, -2.0], vec![0.0, 0.0]);
        let mut opt = Lion::new(0.1, None, None, None);
        {
            let mut refs: Vec<&mut Tensor> = vec![&mut t];
            opt.step(refs.as_mut_slice()).unwrap();
        }
        let got = t.data().as_f64_slice().unwrap().to_vec();
        // sign(0) == 0, no weight decay -> unchanged.
        assert!((got[0] - 1.0).abs() < 1e-12);
        assert!((got[1] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn weight_decay_shrinks_params() {
        let mut t = param(vec![10.0], vec![0.0]);
        let mut opt = Lion::new(0.1, None, None, Some(0.5));
        {
            let mut refs: Vec<&mut Tensor> = vec![&mut t];
            opt.step(refs.as_mut_slice()).unwrap();
        }
        let got = t.data().as_f64_slice().unwrap().to_vec();
        // g = 0 -> sign term 0; decoupled decay: θ -= lr·wd·θ = 10 - 0.1·0.5·10.
        assert!((got[0] - (10.0 - 0.5)).abs() < 1e-12);
    }
}
