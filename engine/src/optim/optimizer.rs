// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::utils::GradientUtils;
use crate::{autograd::TensorId, error::Result, serialization::OptimizerState, tensor::Tensor};
use rustc_hash::FxHashMap;

/// Parameter group for managing different learning rates and settings
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Parameters in this group
    pub params: Vec<TensorId>,
    /// Learning rate for this group
    pub lr: f64,
    /// Weight decay for this group
    pub weight_decay: f64,
    /// Additional group-specific options
    pub options: FxHashMap<String, f64>,
}

impl ParameterGroup {
    /// Create a new parameter group
    pub fn new(params: Vec<TensorId>, lr: f64) -> Self {
        Self {
            params,
            lr,
            weight_decay: 0.0,
            options: FxHashMap::default(),
        }
    }

    /// Create a parameter group with weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Add a custom option to the parameter group
    pub fn with_option(mut self, key: String, value: f64) -> Self {
        self.options.insert(key, value);
        self
    }

    /// Get an option value
    pub fn get_option(&self, key: &str) -> Option<f64> {
        self.options.get(key).copied()
    }
}

/// The parameter-group bookkeeping every optimizer in this module keeps: the
/// groups themselves, a reverse index from parameter to group, and the values
/// to fall back on for a parameter that belongs to no group.
///
/// Each optimizer used to carry the four fields and the three methods over
/// them verbatim -- `rebuild_param_lookup`, `get_param_lr`,
/// `get_param_weight_decay` -- six identical copies whose only difference was
/// the surrounding struct. Sharing them is not only less code: the reverse
/// index is easy to leave stale, and there is now one `push` that cannot.
#[derive(Debug, Clone, Default)]
pub struct ParamGroups {
    groups: Vec<ParameterGroup>,
    /// Fast lookup from parameter id to its group index.
    lookup: FxHashMap<TensorId, usize>,
    /// Learning rate for parameters in no group.
    default_lr: f64,
    /// Weight decay for parameters in no group.
    default_weight_decay: f64,
}

impl ParamGroups {
    /// Empty, with `default_lr` covering every parameter.
    pub fn new(default_lr: f64) -> Self {
        Self {
            groups: Vec::new(),
            lookup: FxHashMap::default(),
            default_lr,
            default_weight_decay: 0.0,
        }
    }

    /// Build from explicit groups. The fallback learning rate is the first
    /// group's, or `fallback_lr` when there are no groups at all.
    pub fn from_groups(groups: Vec<ParameterGroup>, fallback_lr: f64) -> Self {
        let default_lr = groups.first().map(|g| g.lr).unwrap_or(fallback_lr);
        let mut this = Self {
            groups,
            lookup: FxHashMap::default(),
            default_lr,
            default_weight_decay: 0.0,
        };
        this.rebuild_lookup();
        this
    }

    fn rebuild_lookup(&mut self) {
        self.lookup.clear();
        self.lookup
            .reserve(self.groups.iter().map(|g| g.params.len()).sum());
        for (idx, group) in self.groups.iter().enumerate() {
            for &p in &group.params {
                self.lookup.insert(p, idx);
            }
        }
    }

    /// Learning rate for one parameter: its group's, or the default.
    pub fn lr(&self, param_id: TensorId) -> f64 {
        match self.lookup.get(&param_id) {
            Some(&idx) => self.groups[idx].lr,
            None => self.default_lr,
        }
    }

    /// Weight decay for one parameter: its group's, or the default.
    pub fn weight_decay(&self, param_id: TensorId) -> f64 {
        match self.lookup.get(&param_id) {
            Some(&idx) => self.groups[idx].weight_decay,
            None => self.default_weight_decay,
        }
    }

    /// The rate reported by `Optimizer::learning_rate`.
    pub fn default_lr(&self) -> f64 {
        self.default_lr
    }

    /// Set the rate everywhere: the default and every group, which is what
    /// `Optimizer::set_learning_rate` promises.
    pub fn set_lr(&mut self, lr: f64) {
        self.default_lr = lr;
        for group in &mut self.groups {
            group.lr = lr;
        }
    }

    /// The decay applied to parameters in no group.
    pub fn default_weight_decay(&self) -> f64 {
        self.default_weight_decay
    }

    /// Set the decay for parameters in no group. Groups keep their own.
    pub fn set_default_weight_decay(&mut self, weight_decay: f64) {
        self.default_weight_decay = weight_decay;
    }

    pub fn groups(&self) -> &[ParameterGroup] {
        &self.groups
    }

    pub fn groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.groups
    }

    /// Add a group, indexing its parameters as it goes.
    pub fn push(&mut self, group: ParameterGroup) {
        let idx = self.groups.len();
        for &p in &group.params {
            self.lookup.insert(p, idx);
        }
        self.groups.push(group);
    }
}

/// The gradient an optimizer should step on for `param`, or `None` if it has
/// none this iteration and should be skipped.
///
/// The graph is consulted first and `.grad` second: a backward pass leaves the
/// gradient in the graph, and `.grad` is the copy that survives one being
/// released. Every optimizer here opened its loop with these six lines.
pub fn parameter_gradient(param: &Tensor) -> Option<Tensor> {
    if let Some(g) = crate::autograd::get_gradient(param) {
        Some(g)
    } else {
        param.grad().map(|g| (**g).clone())
    }
}

/// Reject a gradient that cannot be applied to its parameter elementwise.
///
/// Every per-parameter update below indexes the two buffers in lockstep, so a
/// mismatch here would be a wrong answer rather than a slow one.
pub fn check_param_grad_match(param: &Tensor, grad: &Tensor) -> Result<()> {
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
    Ok(())
}

/// Implements the seven [`Optimizer`] methods that are pure bookkeeping —
/// identical in every optimizer here, and differing only in which fields hold
/// the groups and the step count.
///
/// `$groups` names a [`ParamGroups`] field, `$step_count` a `usize` one.
#[macro_export]
macro_rules! delegate_optimizer_bookkeeping {
    ($groups:ident, $step_count:ident) => {
        fn zero_grad(
            &self,
            parameters: &mut [&mut $crate::tensor::Tensor],
            set_to_none: bool,
        ) -> $crate::error::Result<()> {
            for param in parameters.iter_mut() {
                param.zero_grad(set_to_none);
            }
            Ok(())
        }

        fn learning_rate(&self) -> f64 {
            self.$groups.default_lr()
        }

        fn set_learning_rate(&mut self, lr: f64) {
            self.$groups.set_lr(lr);
        }

        fn param_groups(&self) -> &[$crate::optim::ParameterGroup] {
            self.$groups.groups()
        }

        fn param_groups_mut(&mut self) -> &mut [$crate::optim::ParameterGroup] {
            self.$groups.groups_mut()
        }

        fn add_param_group(
            &mut self,
            group: $crate::optim::ParameterGroup,
        ) -> $crate::error::Result<()> {
            self.$groups.push(group);
            Ok(())
        }

        fn step_count(&self) -> usize {
            self.$step_count
        }
    };
}

/// Gradient clipping configuration
#[derive(Debug, Clone, Default)]
pub enum GradientClipping {
    /// No gradient clipping
    #[default]
    None,
    /// Clip gradients by norm
    ByNorm { max_norm: f64 },
    /// Clip gradients by value
    ByValue { min_value: f64, max_value: f64 },
}

/// Learning rate scheduler interface
pub trait LearningRateScheduler: Send + Sync {
    /// Get the learning rate for the current step
    fn get_lr(&self, step: usize, base_lr: f64) -> f64;

    /// Update scheduler state (if needed)
    fn step(&mut self) {}
}

/// Constant learning rate scheduler
#[derive(Debug, Clone)]
pub struct ConstantLR;

impl LearningRateScheduler for ConstantLR {
    fn get_lr(&self, _step: usize, base_lr: f64) -> f64 {
        base_lr
    }
}

/// Step learning rate scheduler
#[derive(Debug, Clone)]
pub struct StepLR {
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f64) -> Self {
        Self { step_size, gamma }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if self.step_size == 0 {
            return base_lr;
        }
        let decay_factor = self.gamma.powi((step / self.step_size) as i32);
        base_lr * decay_factor
    }
}

/// Exponential learning rate scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        base_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing learning rate scheduler
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f64,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f64) -> Self {
        Self { t_max, eta_min }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if self.t_max == 0 {
            return base_lr;
        }

        let t = step.min(self.t_max) as f64;
        let t_max = self.t_max as f64;

        // Standard cosine annealing formula
        // At t=0: cos(0) = 1, lr = base_lr
        // At t=t_max/2: cos(π/2) = 0, lr = (base_lr + eta_min)/2
        // At t=t_max: cos(π) = -1, lr = eta_min
        self.eta_min
            + (base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * t / t_max).cos()) / 2.0
    }
}

/// Trait for optimization algorithms
pub trait Optimizer: Send + Sync {
    /// Perform one optimization step
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;

    /// Zero out gradients of parameters
    fn zero_grad(&self, parameters: &mut [&mut Tensor], set_to_none: bool) -> Result<()>;

    /// Get the learning rate (for single parameter group optimizers)
    fn learning_rate(&self) -> f64;

    /// Set the learning rate (for single parameter group optimizers)
    fn set_learning_rate(&mut self, lr: f64);

    /// Get parameter groups
    fn param_groups(&self) -> &[ParameterGroup] {
        // Default implementation for backward compatibility
        &[]
    }

    /// Get mutable parameter groups
    fn param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        // Default implementation for backward compatibility
        &mut []
    }

    /// Add a parameter group
    fn add_param_group(&mut self, _group: ParameterGroup) -> Result<()> {
        // Default implementation for backward compatibility
        Ok(())
    }

    /// Get the current step count
    fn step_count(&self) -> usize {
        0
    }

    /// Snapshot everything needed to resume training exactly where it stopped.
    ///
    /// `parameters` must be the same list, in the same order, that `step` is
    /// called with: per-parameter buffers are keyed by position, because the
    /// [`TensorId`]s they are keyed by internally do not survive a reload.
    fn state_dict(&self, parameters: &[&Tensor]) -> Result<OptimizerState> {
        let _ = parameters;
        Err(crate::error::MinitensorError::not_implemented(
            "this optimizer does not implement state_dict",
        ))
    }

    /// Restore a snapshot from [`Self::state_dict`].
    fn load_state_dict(&mut self, parameters: &[&Tensor], state: &OptimizerState) -> Result<()> {
        let _ = (parameters, state);
        Err(crate::error::MinitensorError::not_implemented(
            "this optimizer does not implement load_state_dict",
        ))
    }

    /// Apply gradient clipping to parameters
    fn clip_gradients(
        &self,
        parameters: &mut [&mut Tensor],
        clipping: &GradientClipping,
    ) -> Result<()> {
        match clipping {
            GradientClipping::None => Ok(()),
            GradientClipping::ByNorm { max_norm } => self.clip_grad_norm(parameters, *max_norm),
            GradientClipping::ByValue {
                min_value,
                max_value,
            } => self.clip_grad_value(parameters, *min_value, *max_value),
        }
    }

    /// Clip gradients by norm
    fn clip_grad_norm(&self, parameters: &mut [&mut Tensor], max_norm: f64) -> Result<()> {
        GradientUtils::clip_grad_norm(parameters, max_norm).map(|_| ())
    }

    /// Clip gradients by value
    fn clip_grad_value(
        &self,
        parameters: &mut [&mut Tensor],
        min_value: f64,
        max_value: f64,
    ) -> Result<()> {
        GradientUtils::clip_grad_value(parameters, min_value, max_value)
    }

    /// Set the learning rate from `scheduler` for the current step.
    ///
    /// `base_lr` must be the *initial* learning rate, not the current one:
    /// the schedulers in this module are stateless functions of
    /// `(step, base_lr)`, so feeding them the already-decayed rate compounds
    /// the decay on every call (`ExponentialLR` would apply
    /// `gamma^(1 + 2 + … + n)` instead of `gamma^n`). Callers that vary the
    /// base rate per group should drive `set_learning_rate` /
    /// [`ParameterGroup::lr`] directly.
    fn apply_lr_scheduler(&mut self, scheduler: &dyn LearningRateScheduler, base_lr: f64) {
        let step = self.step_count();
        let new_lr = scheduler.get_lr(step, base_lr);

        if self.param_groups().is_empty() {
            self.set_learning_rate(new_lr);
        } else {
            for group in self.param_groups_mut() {
                group.lr = new_lr;
            }
        }
    }
}

#[cfg(test)]
mod param_groups_tests {
    use super::*;
    use crate::autograd::TensorId;

    fn ids(n: usize) -> Vec<TensorId> {
        (0..n).map(|_| TensorId::new()).collect()
    }

    /// A parameter's group decides its rate and decay; one in no group falls
    /// back to the defaults. This is the lookup every optimizer used to keep
    /// its own copy of.
    #[test]
    fn lookup_prefers_the_group_then_the_default() {
        let p = ids(3);
        let groups = vec![
            ParameterGroup::new(vec![p[0]], 0.1).with_weight_decay(0.01),
            ParameterGroup::new(vec![p[1]], 0.2).with_weight_decay(0.02),
        ];
        let mut pg = ParamGroups::from_groups(groups, 9.9);

        assert_eq!(pg.lr(p[0]), 0.1);
        assert_eq!(pg.lr(p[1]), 0.2);
        assert_eq!(pg.weight_decay(p[0]), 0.01);
        assert_eq!(pg.weight_decay(p[1]), 0.02);

        // `p[2]` is in no group: defaults. The fallback rate is the first
        // group's, not the `fallback_lr` argument, which only applies when
        // there are no groups at all.
        assert_eq!(pg.lr(p[2]), 0.1);
        assert_eq!(pg.weight_decay(p[2]), 0.0);
        pg.set_default_weight_decay(0.5);
        assert_eq!(pg.weight_decay(p[2]), 0.5);
        // and a grouped parameter is unaffected by the default
        assert_eq!(pg.weight_decay(p[0]), 0.01);

        assert_eq!(ParamGroups::from_groups(Vec::new(), 9.9).lr(p[2]), 9.9);
    }

    /// A group added after construction is indexed as it goes in -- the
    /// failure mode of a hand-maintained reverse index is that it is not, and
    /// the parameter silently keeps stepping at the default rate.
    #[test]
    fn a_pushed_group_is_immediately_findable() {
        let p = ids(2);
        let mut pg = ParamGroups::new(0.5);
        assert_eq!(pg.lr(p[0]), 0.5);

        pg.push(ParameterGroup::new(vec![p[0]], 0.01).with_weight_decay(0.7));
        assert_eq!(pg.lr(p[0]), 0.01);
        assert_eq!(pg.weight_decay(p[0]), 0.7);
        assert_eq!(pg.lr(p[1]), 0.5, "an unrelated parameter keeps the default");
        assert_eq!(pg.groups().len(), 1);
    }

    /// `set_learning_rate` moves every group as well as the default, which is
    /// what makes it usable on an optimizer that has groups.
    #[test]
    fn setting_the_rate_moves_every_group() {
        let p = ids(2);
        let mut pg = ParamGroups::from_groups(
            vec![
                ParameterGroup::new(vec![p[0]], 0.1),
                ParameterGroup::new(vec![p[1]], 0.2),
            ],
            0.0,
        );
        pg.set_lr(0.03);
        assert_eq!(pg.default_lr(), 0.03);
        assert_eq!(pg.lr(p[0]), 0.03);
        assert_eq!(pg.lr(p[1]), 0.03);
        // ...and leaves the per-group decay alone
        pg.groups_mut()[0].weight_decay = 0.9;
        assert_eq!(pg.weight_decay(p[0]), 0.9);
    }
}
