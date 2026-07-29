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
    /// Parameter groups with per-group learning rate / weight decay.
    param_groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index.
    param_lookup: FxHashMap<TensorId, usize>,
    /// Default learning rate (single-group case).
    default_lr: f64,
    /// Momentum interpolation coefficient for the update direction.
    beta1: f64,
    /// Momentum EMA coefficient for the stored buffer.
    beta2: f64,
    /// Decoupled weight decay coefficient.
    weight_decay: f64,
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
            param_groups: Vec::new(),
            param_lookup: FxHashMap::default(),
            default_lr: learning_rate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.99),
            weight_decay: weight_decay.unwrap_or(0.0),
            m: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Rebuild internal parameter lookup table.
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

    /// Create a new Lion optimizer with explicit parameter groups.
    pub fn with_param_groups(param_groups: Vec<ParameterGroup>, beta1: f64, beta2: f64) -> Self {
        let default_lr = param_groups.first().map(|g| g.lr).unwrap_or(1e-4);
        let mut optimizer = Self {
            param_groups,
            param_lookup: FxHashMap::default(),
            default_lr,
            beta1,
            beta2,
            weight_decay: 0.0,
            m: FxHashMap::default(),
            step_count: 0,
            gradient_clipping: GradientClipping::default(),
        };
        optimizer.rebuild_param_lookup();
        optimizer
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
        self.weight_decay
    }

    fn get_param_lr(&self, param_id: TensorId) -> f64 {
        if let Some(&idx) = self.param_lookup.get(&param_id) {
            self.param_groups[idx].lr
        } else {
            self.default_lr
        }
    }

    fn get_param_weight_decay(&self, param_id: TensorId) -> f64 {
        if let Some(&idx) = self.param_lookup.get(&param_id) {
            self.param_groups[idx].weight_decay
        } else {
            self.weight_decay
        }
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
                    p.par_chunks_mut(PAR_CHUNK)
                        .zip(g.par_chunks(PAR_CHUNK))
                        .zip(m_buf.par_chunks_mut(PAR_CHUNK))
                        .for_each(|((p, g), m)| step_chunk(p, g, m));
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
            self.apply_lion_update(param, &grad, lr, weight_decay)?;
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
