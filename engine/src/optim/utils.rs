// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::optimizer::LearningRateScheduler;
use crate::{
    autograd::{self, TensorId},
    error::{MinitensorError, Result},
    serialization::OptimizerState,
    tensor::Tensor,
};
use rustc_hash::FxHashMap;

/// Utility functions for gradient operations.
///
/// Every entry point here resolves a parameter's gradient exactly the way
/// `Optimizer::step` does — the autograd graph first, then the tensor-local
/// `.grad` — so clipping acts on the gradient the step will actually consume.
/// (Reading only `.grad` made clipping a silent no-op after the usual
/// `loss.backward()`, which stores gradients in the graph.)
///
/// The per-parameter loops are deliberately sequential: the autograd graph is
/// thread-local, so a rayon worker would look at an empty graph. Parallelism
/// lives in the per-element loops below, where the work actually is.
pub struct GradientUtils;

/// Element count above which the gradient element loops go parallel.
///
/// These loops -- a sum of squares, a scale, a clamp -- are as cheap per
/// element as a `relu`, so they use the same threshold as the cheap unary
/// kernels rather than a second, lower copy of the number. See
/// [`crate::ops::map::PAR_THRESHOLD`] for the measurements: below it, waking
/// rayon's parked workers costs more than the whole loop.
use crate::ops::map::PAR_THRESHOLD as GRAD_PAR_THRESHOLD;
use crate::ops::map::{PAR_CHUNK, par_out_chunks};

/// Sum of squares of a float slice, parallel above the threshold.
fn sum_squares<T: Copy + Send + Sync + Into<f64>>(values: &[T]) -> f64 {
    let square = |v: &T| {
        let v: f64 = (*v).into();
        v * v
    };
    if values.len() < GRAD_PAR_THRESHOLD {
        values.iter().map(square).sum()
    } else {
        // Chunked and folded in index order: a gradient norm that shifts in
        // its last bits between runs makes `clip_grad_norm_` scale by a
        // slightly different factor each time, which is exactly the kind of
        // irreproducibility a seeded training run is supposed to rule out.
        crate::ops::util::deterministic_par_sum(values, 8192, |chunk| {
            chunk.iter().map(square).sum::<f64>()
        })
    }
}

/// Apply `op` to every element, parallel above the threshold.
fn map_in_place<T: Copy + Send + Sync>(values: &mut [T], op: impl Fn(T) -> T + Send + Sync) {
    if values.len() < GRAD_PAR_THRESHOLD {
        for v in values.iter_mut() {
            *v = op(*v);
        }
    } else {
        par_out_chunks(values, PAR_CHUNK, &|_, chunk| {
            for v in chunk.iter_mut() {
                *v = op(*v);
            }
        });
    }
}

/// The gradient `Optimizer::step` would use for this parameter, if any.
fn resolve_gradient(param: &Tensor) -> Option<Tensor> {
    autograd::get_gradient(param).or_else(|| param.grad().map(|g| (**g).clone()))
}

/// Read-modify-write a parameter's gradient in place, wherever it lives.
///
/// A graph-stored gradient is taken out of the map before being mutated so
/// the write does not trigger copy-on-write, then put back.
fn with_gradient_mut(param: &mut Tensor, transform: impl Fn(&mut Tensor)) {
    if let Some(mut grad) = autograd::clear_gradient(param) {
        transform(&mut grad);
        autograd::set_gradient(param, grad);
        return;
    }
    if let Some(grad) = param.grad_mut() {
        transform(grad);
    }
}

impl GradientUtils {
    fn compute_grad_norm_value(parameters: &[&Tensor]) -> f64 {
        let total_sq_norm: f64 = parameters
            .iter()
            .filter_map(|param| resolve_gradient(param))
            .map(|grad| match grad.dtype() {
                crate::tensor::DataType::Float32 => {
                    sum_squares(grad.data().as_f32_slice().unwrap_or_default())
                }
                crate::tensor::DataType::Float64 => {
                    sum_squares(grad.data().as_f64_slice().unwrap_or_default())
                }
                _ => 0.0,
            })
            .sum();
        total_sq_norm.sqrt()
    }

    /// Compute the L2 norm of gradients across all parameters
    pub fn compute_grad_norm(parameters: &[&Tensor]) -> Result<f64> {
        Ok(Self::compute_grad_norm_value(parameters))
    }

    /// Apply gradient clipping by norm to a set of parameters
    pub fn clip_grad_norm(parameters: &mut [&mut Tensor], max_norm: f64) -> Result<f64> {
        let total_norm = {
            let refs: Vec<&Tensor> = parameters.iter().map(|p| &**p).collect();
            Self::compute_grad_norm_value(&refs)
        };

        if total_norm > max_norm {
            let clip_coef = max_norm / (total_norm + 1e-6);
            let coef_f32 = clip_coef as f32;
            for param in parameters.iter_mut() {
                with_gradient_mut(param, |grad| match grad.dtype() {
                    crate::tensor::DataType::Float32 => {
                        if let Some(g) = grad.data_mut().as_f32_slice_mut() {
                            map_in_place(g, |v| v * coef_f32);
                        }
                    }
                    crate::tensor::DataType::Float64 => {
                        if let Some(g) = grad.data_mut().as_f64_slice_mut() {
                            map_in_place(g, |v| v * clip_coef);
                        }
                    }
                    _ => {}
                });
            }
        }

        Ok(total_norm)
    }

    /// Apply gradient clipping by value to a set of parameters
    pub fn clip_grad_value(
        parameters: &mut [&mut Tensor],
        min_value: f64,
        max_value: f64,
    ) -> Result<()> {
        let min_f32 = min_value as f32;
        let max_f32 = max_value as f32;
        for param in parameters.iter_mut() {
            with_gradient_mut(param, |grad| match grad.dtype() {
                crate::tensor::DataType::Float32 => {
                    if let Some(g) = grad.data_mut().as_f32_slice_mut() {
                        map_in_place(g, |v| v.clamp(min_f32, max_f32));
                    }
                }
                crate::tensor::DataType::Float64 => {
                    if let Some(g) = grad.data_mut().as_f64_slice_mut() {
                        map_in_place(g, |v| v.clamp(min_value, max_value));
                    }
                }
                _ => {}
            });
        }

        Ok(())
    }

    /// Check if any parameters have gradients
    pub fn has_gradients(parameters: &[&Tensor]) -> bool {
        parameters
            .iter()
            .any(|param| resolve_gradient(param).is_some())
    }

    /// Count the number of parameters with gradients
    pub fn count_parameters_with_gradients(parameters: &[&Tensor]) -> usize {
        parameters
            .iter()
            .filter(|param| resolve_gradient(param).is_some())
            .count()
    }
}

/// Learning rate scheduler utilities
pub struct SchedulerUtils;

impl SchedulerUtils {
    /// Create a linear warmup scheduler that increases learning rate linearly
    pub fn linear_warmup(warmup_steps: usize) -> LinearWarmupScheduler {
        LinearWarmupScheduler::new(warmup_steps)
    }

    /// Create a polynomial decay scheduler
    pub fn polynomial_decay(
        decay_steps: usize,
        end_lr: f64,
        power: f64,
    ) -> PolynomialDecayScheduler {
        PolynomialDecayScheduler::new(decay_steps, end_lr, power)
    }

    /// Create a multi-step scheduler with multiple decay points
    pub fn multi_step(milestones: Vec<usize>, gamma: f64) -> MultiStepScheduler {
        MultiStepScheduler::new(milestones, gamma)
    }
}

/// Linear warmup learning rate scheduler
#[derive(Debug, Clone)]
pub struct LinearWarmupScheduler {
    warmup_steps: usize,
}

impl LinearWarmupScheduler {
    pub fn new(warmup_steps: usize) -> Self {
        Self { warmup_steps }
    }
}

impl LearningRateScheduler for LinearWarmupScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if step < self.warmup_steps {
            base_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            base_lr
        }
    }
}

/// Polynomial decay learning rate scheduler
#[derive(Debug, Clone)]
pub struct PolynomialDecayScheduler {
    decay_steps: usize,
    end_lr: f64,
    power: f64,
}

impl PolynomialDecayScheduler {
    pub fn new(decay_steps: usize, end_lr: f64, power: f64) -> Self {
        Self {
            decay_steps,
            end_lr,
            power,
        }
    }
}

impl LearningRateScheduler for PolynomialDecayScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        if step >= self.decay_steps {
            return self.end_lr;
        }

        let decay_factor = (1.0 - step as f64 / self.decay_steps as f64).powf(self.power);
        (base_lr - self.end_lr) * decay_factor + self.end_lr
    }
}

/// Multi-step learning rate scheduler
#[derive(Debug, Clone)]
pub struct MultiStepScheduler {
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepScheduler {
    pub fn new(mut milestones: Vec<usize>, gamma: f64) -> Self {
        milestones.sort_unstable();
        Self { milestones, gamma }
    }
}

impl LearningRateScheduler for MultiStepScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        let decay_count = self
            .milestones
            .iter()
            .filter(|&&milestone| step >= milestone)
            .count();
        base_lr * self.gamma.powi(decay_count as i32)
    }
}

/// Composite scheduler that combines multiple schedulers
pub struct CompositeScheduler {
    schedulers: Vec<(Box<dyn LearningRateScheduler>, usize)>, // (scheduler, start_step)
}

impl CompositeScheduler {
    pub fn new() -> Self {
        Self {
            schedulers: Vec::new(),
        }
    }

    /// Add a scheduler that starts at a specific step
    pub fn add_scheduler(
        mut self,
        scheduler: Box<dyn LearningRateScheduler>,
        start_step: usize,
    ) -> Self {
        self.schedulers.push((scheduler, start_step));
        // Sort by start step
        self.schedulers.sort_by_key(|(_, start)| *start);
        self
    }
}

impl Default for CompositeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningRateScheduler for CompositeScheduler {
    fn get_lr(&self, step: usize, base_lr: f64) -> f64 {
        // Find the most recent scheduler that should be active
        let mut current_lr = base_lr;

        for (scheduler, start_step) in &self.schedulers {
            if step >= *start_step {
                current_lr = scheduler.get_lr(step - start_step, base_lr);
            } else {
                break;
            }
        }

        current_lr
    }
}

/// Copy a per-parameter buffer map into an [`OptimizerState`], converting the
/// [`TensorId`] keys into the positions the state is keyed by.
///
/// A parameter with no entry is skipped rather than written as zeros: every
/// optimizer here allocates its buffers on a parameter's first step, so a
/// parameter that has not been stepped yet genuinely has no state, and writing
/// zeros would make a resumed run start from a "first step already taken"
/// position it was never in.
pub(crate) fn save_param_buffers(
    state: &mut OptimizerState,
    name: &str,
    buffers: &FxHashMap<TensorId, Tensor>,
    parameters: &[&Tensor],
) -> Result<()> {
    for (slot, param) in parameters.iter().enumerate() {
        if let Some(tensor) = buffers.get(&param.id()) {
            state.insert_buffer(slot, name, tensor)?;
        }
    }
    Ok(())
}

/// Inverse of [`save_param_buffers`], rekeying by the ids the *current*
/// parameters carry.
///
/// The map is cleared first. Leaving stale entries would keep the state of
/// whatever parameters the optimizer had stepped before the load, which for a
/// reused optimizer is state from a different run.
pub(crate) fn load_param_buffers(
    state: &OptimizerState,
    name: &str,
    buffers: &mut FxHashMap<TensorId, Tensor>,
    parameters: &[&Tensor],
) -> Result<()> {
    buffers.clear();
    for (slot, param) in parameters.iter().enumerate() {
        if let Some(tensor) = state.take_buffer(slot, name, Some(param.device()))? {
            if tensor.shape().dims() != param.shape().dims() {
                return Err(MinitensorError::invalid_argument_with_suggestion(
                    format!(
                        "optimizer state for parameter {slot} has shape {:?}, but that \
                         parameter is {:?}",
                        tensor.shape().dims(),
                        param.shape().dims()
                    ),
                    "Per-parameter state is matched by position, so the optimizer must \
                     be constructed over the same parameters in the same order as when \
                     it was saved",
                ));
            }
            buffers.insert(param.id(), tensor);
        }
    }
    Ok(())
}

/// The parameter walk every [`Optimizer::step`] shares.
///
/// Hands `update` each parameter that both wants a gradient and has one, in
/// the order they were given. Everything an optimizer does that is *per step*
/// rather than per parameter -- clipping, counting, a momentum schedule --
/// stays in its own `step`, before this is called.
///
/// It exists because the two `continue`s are a policy, not boilerplate: a
/// parameter with `requires_grad` false is skipped, and one whose gradient was
/// never computed is skipped rather than treated as a zero, which would decay
/// its weights on a step it took no part in. That policy was written out nine
/// times, once per optimizer, where nine copies could drift apart.
///
/// [`Optimizer::step`]: super::optimizer::Optimizer::step
pub(crate) fn step_each_parameter<F>(
    parameters: &mut [&mut crate::tensor::Tensor],
    mut update: F,
) -> Result<()>
where
    F: FnMut(&mut crate::tensor::Tensor, &crate::tensor::Tensor) -> Result<()>,
{
    for param in parameters.iter_mut() {
        if !param.requires_grad() {
            continue;
        }
        let Some(grad) = super::optimizer::parameter_gradient(param) else {
            continue;
        };
        update(param, &grad)?;
    }
    Ok(())
}

/// One float width's arm of [`fixed_state_update!`]. Not called directly.
///
/// Split out only because a macro cannot easily expand a nested `macro_rules!`
/// with its own metavariables; this is the body [`fixed_state_update!`]
/// instantiates once per width.
macro_rules! fixed_state_arm {
    (
        $ty:ty, $read:ident, $write:ident, $name:literal,
        $param:expr, $grad:expr, [$($state:expr),+ $(,)?],
        [$($scalar:ident = $value:expr),* $(,)?],
        |$p:ident, $g:ident, $i:ident, [$($slot:ident),+ $(,)?]| $body:block
    ) => {{
        // Once per call rather than once per element.
        $(let $scalar: $ty = $value as $ty;)*

        let slice_error = |what: &str| {
            $crate::error::MinitensorError::internal_error(format!(
                concat!($name, ": failed to read the {} as ", stringify!($ty)),
                what
            ))
        };
        let param_buffer = $param
            .data_mut()
            .$write()
            .ok_or_else(|| slice_error("parameter"))?;
        let grad_buffer = $grad.data().$read().ok_or_else(|| slice_error("gradient"))?;
        $(
            let $slot = $state
                .data_mut()
                .$write()
                .ok_or_else(|| slice_error(stringify!($slot)))?;
        )+

        let step = |$p: &mut [$ty], $g: &[$ty], state: &mut [&mut [$ty]]| {
            let [$($slot),+] = state else {
                unreachable!(concat!($name, " passes its own state buffers"))
            };
            for $i in 0..$p.len() {
                $body
            }
        };

        // Below the threshold rayon's split overhead dwarfs the arithmetic, so
        // stay on the calling thread. See `ops::map::PAR_THRESHOLD`.
        let mut state = [$($slot),+];
        if param_buffer.len() < $crate::ops::map::PAR_THRESHOLD {
            step(param_buffer, grad_buffer, &mut state);
        } else {
            $crate::ops::map::par_param_update(
                param_buffer,
                grad_buffer,
                &mut state,
                $crate::ops::map::PAR_CHUNK,
                &step,
            );
        }
        Ok(())
    }};
}

/// Run one optimizer's per-element update at whichever float width the
/// parameter carries, sequentially below the element threshold and across
/// rayon's workers above it.
///
/// The update body is written once and instantiated at both widths, which is
/// what makes this a macro: the body has to be generic over the width, and a
/// closure cannot be. So it must not name `f32` or `f64` -- write `0.5`, not
/// `0.5f64` -- and its scalars arrive through the `[name = value]` list,
/// converted to the working width once per call rather than once per element.
///
/// This covers the optimizers whose state is the same buffers on every step.
/// The ones that came before it take a set that varies at runtime -- RMSprop's
/// momentum and centering buffers, Adam's `amsgrad` maximum, SGD's momentum --
/// and the code that hands rayon only the buffers actually in use is the
/// interesting part of those files, not boilerplate to be shared away.
macro_rules! fixed_state_update {
    (
        $name:literal,
        $param:expr, $grad:expr, [$($state:expr),+ $(,)?],
        [$($scalar:ident = $value:expr),* $(,)?],
        |$p:ident, $g:ident, $i:ident, [$($slot:ident),+ $(,)?]| $body:block
    ) => {{
        let param: &mut $crate::tensor::Tensor = $param;
        let grad: &$crate::tensor::Tensor = $grad;
        match param.dtype() {
            $crate::tensor::DataType::Float32 => $crate::optim::utils::fixed_state_arm!(
                f32, as_f32_slice, as_f32_slice_mut, $name,
                param, grad, [$($state),+], [$($scalar = $value),*],
                |$p, $g, $i, [$($slot),+]| $body
            ),
            $crate::tensor::DataType::Float64 => $crate::optim::utils::fixed_state_arm!(
                f64, as_f64_slice, as_f64_slice_mut, $name,
                param, grad, [$($state),+], [$($scalar = $value),*],
                |$p, $g, $i, [$($slot),+]| $body
            ),
            other => Err($crate::error::MinitensorError::invalid_operation(format!(
                concat!($name, " requires floating point parameters, got {}"),
                other
            ))),
        }
    }};
}

pub(crate) use {fixed_state_arm, fixed_state_update};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, Tensor},
    };

    #[test]
    fn test_gradient_utils_has_gradients() {
        let shape = Shape::new(vec![2, 2]);
        let tensor1 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let mut tensor2 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        // Set gradient on tensor2
        let grad = Tensor::ones(shape, DataType::Float32, Device::cpu(), false);
        tensor2.set_grad(Some(grad));

        let params = vec![&tensor1, &tensor2];
        assert!(GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 1);
    }

    #[test]
    fn test_gradient_utils_no_gradients() {
        let shape = Shape::new(vec![2, 2]);
        let tensor1 = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let tensor2 = Tensor::zeros(shape, DataType::Float64, Device::cpu(), true);

        let params = vec![&tensor1, &tensor2];
        assert!(!GradientUtils::has_gradients(&params));
        assert_eq!(GradientUtils::count_parameters_with_gradients(&params), 0);
        assert_eq!(GradientUtils::compute_grad_norm(&params).unwrap(), 0.0);
    }

    #[test]
    fn test_clip_grad_norm_clips_float64_and_skips_non_float() {
        let mut float_param =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut int_param =
            Tensor::zeros(Shape::new(vec![2]), DataType::Int64, Device::cpu(), true);

        let mut float_grad =
            Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        float_grad
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[6.0, 8.0]);
        let mut int_grad = Tensor::ones(Shape::new(vec![2]), DataType::Int64, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i64_slice_mut()
            .unwrap()
            .copy_from_slice(&[7, -3]);

        float_param.set_grad(Some(float_grad));
        int_param.set_grad(Some(int_grad));

        let mut params = vec![&mut float_param, &mut int_param];
        let original_norm = GradientUtils::clip_grad_norm(&mut params, 5.0).unwrap();
        assert!((original_norm - 10.0).abs() < 1e-10);

        let scaled_float = params[0].grad().unwrap().data().as_f64_slice().unwrap();
        assert!((scaled_float[0] - 3.0).abs() < 1e-6);
        assert!((scaled_float[1] - 4.0).abs() < 1e-6);

        let unchanged_int = params[1].grad().unwrap().data().as_i64_slice().unwrap();
        assert_eq!(unchanged_int, &[7, -3]);
    }

    #[test]
    fn test_clip_grad_value_no_gradients_is_noop() {
        let mut p1 = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut p2 = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);

        let mut params = vec![&mut p1, &mut p2];
        GradientUtils::clip_grad_value(&mut params, -1.0, 1.0).unwrap();

        assert!(params[0].grad().is_none());
        assert!(params[1].grad().is_none());
    }

    #[test]
    fn test_clip_grad_norm_empty_parameter_list() {
        let mut params: Vec<&mut Tensor> = Vec::new();
        let norm = GradientUtils::clip_grad_norm(&mut params, 1.0).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_value_empty_parameter_list() {
        let mut params: Vec<&mut Tensor> = Vec::new();
        GradientUtils::clip_grad_value(&mut params, -0.5, 0.5).unwrap();
    }
    #[test]
    fn test_linear_warmup_zero_steps_returns_base_lr() {
        let scheduler = LinearWarmupScheduler::new(0);
        assert_eq!(scheduler.get_lr(0, 0.2), 0.2);
        assert_eq!(scheduler.get_lr(10, 0.2), 0.2);
    }

    #[test]
    fn test_polynomial_decay_zero_decay_steps_returns_end_lr() {
        let scheduler = PolynomialDecayScheduler::new(0, 0.05, 2.0);
        assert_eq!(scheduler.get_lr(0, 0.5), 0.05);
        assert_eq!(scheduler.get_lr(5, 0.5), 0.05);
    }

    #[test]
    fn test_composite_scheduler_sorts_start_steps() {
        let base_lr = 1.0;
        let scheduler = CompositeScheduler::new()
            .add_scheduler(Box::new(MultiStepScheduler::new(vec![1], 0.5)), 10)
            .add_scheduler(Box::new(LinearWarmupScheduler::new(5)), 0);

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(4, base_lr), 0.8);
        assert_eq!(scheduler.get_lr(10, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(11, base_lr), 0.5);
    }

    #[test]
    fn test_linear_warmup_scheduler() {
        let scheduler = LinearWarmupScheduler::new(10);
        let base_lr = 0.1;

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(5, base_lr), 0.05);
        assert_eq!(scheduler.get_lr(10, base_lr), 0.1);
        assert_eq!(scheduler.get_lr(15, base_lr), 0.1);
    }

    #[test]
    fn test_polynomial_decay_scheduler() {
        let scheduler = PolynomialDecayScheduler::new(100, 0.01, 2.0);
        let base_lr = 0.1;

        assert_eq!(scheduler.get_lr(0, base_lr), base_lr);
        assert!(scheduler.get_lr(50, base_lr) > 0.01);
        assert!(scheduler.get_lr(50, base_lr) < base_lr);
        assert_eq!(scheduler.get_lr(100, base_lr), 0.01);
        assert_eq!(scheduler.get_lr(150, base_lr), 0.01);
    }

    #[test]
    fn test_multi_step_scheduler() {
        let scheduler = MultiStepScheduler::new(vec![30, 60, 90], 0.1);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(29, base_lr), 1.0);
        assert!((scheduler.get_lr(30, base_lr) - 0.1).abs() < 1e-10);
        assert!((scheduler.get_lr(60, base_lr) - 0.01).abs() < 1e-10);
        assert!((scheduler.get_lr(90, base_lr) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_step_lr_scheduler() {
        use super::super::optimizer::StepLR;
        let scheduler = StepLR::new(10, 0.5);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(9, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(10, base_lr), 0.5);
        assert_eq!(scheduler.get_lr(20, base_lr), 0.25);
    }

    #[test]
    fn test_step_lr_zero_step_size_defaults() {
        use super::super::optimizer::StepLR;
        let scheduler = StepLR::new(0, 0.5);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 1.0);
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        use super::super::optimizer::ExponentialLR;
        let scheduler = ExponentialLR::new(0.9);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 0.9);
        assert!((scheduler.get_lr(2, base_lr) - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        use super::super::optimizer::CosineAnnealingLR;
        let scheduler = CosineAnnealingLR::new(100, 0.0);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        // At step 50 of 100, we have cos(π * 50/100) = cos(π/2) = 0
        // So lr = 0 + (1-0) * (1+0)/2 = 0.5
        assert!((scheduler.get_lr(50, base_lr) - 0.5).abs() < 1e-10);
        // At step 100, lr should reach eta_min.
        assert_eq!(scheduler.get_lr(100, base_lr), 0.0);
        // After t_max, lr should stay at eta_min.
        assert_eq!(scheduler.get_lr(150, base_lr), 0.0);
    }

    #[test]
    fn test_cosine_annealing_zero_t_max_defaults() {
        use super::super::optimizer::CosineAnnealingLR;
        let scheduler = CosineAnnealingLR::new(0, 0.1);
        let base_lr = 1.0;

        assert_eq!(scheduler.get_lr(0, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(1, base_lr), 1.0);
    }

    #[test]
    fn test_compute_grad_norm_float32_and_float64() {
        let shape = Shape::new(vec![2]);
        let mut float32_param =
            Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);
        let mut float64_param = Tensor::zeros(shape, DataType::Float64, Device::cpu(), true);

        let mut grad_f32 =
            Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad_f32
            .data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        let mut grad_f64 =
            Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        grad_f64
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[12.0, 0.0]);

        float32_param.set_grad(Some(grad_f32));
        float64_param.set_grad(Some(grad_f64));

        let params = vec![&float32_param, &float64_param];
        let norm = GradientUtils::compute_grad_norm(&params).unwrap();
        assert!((norm - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_grad_norm_ignores_non_float_gradients() {
        let mut param = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut int_grad = Tensor::ones(Shape::new(vec![2]), DataType::Int32, Device::cpu(), false);
        int_grad
            .data_mut()
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&[10, -10]);
        param.set_grad(Some(int_grad));

        let params = vec![&param];
        let norm = GradientUtils::compute_grad_norm(&params).unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_clip_grad_norm_scales_down_gradients() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        p.set_grad(Some(grad));

        let mut params = vec![&mut p];
        let original_norm = GradientUtils::clip_grad_norm(&mut params, 2.5).unwrap();
        assert!((original_norm - 5.0).abs() < 1e-10);

        let scaled = params[0].grad().unwrap().data().as_f32_slice().unwrap();
        assert!((scaled[0] - 1.5).abs() < 1e-3);
        assert!((scaled[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_clip_grad_norm_handles_parameters_without_gradients_in_clip_pass() {
        let mut with_grad =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut without_grad =
            Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);

        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[10.0, 0.0]);
        with_grad.set_grad(Some(grad));

        let mut params = vec![&mut with_grad, &mut without_grad];
        let norm = GradientUtils::clip_grad_norm(&mut params, 1.0).unwrap();
        assert!(norm > 1.0);
        assert!(params[1].grad().is_none());
    }
    #[test]
    fn test_clip_grad_norm_noop_when_within_threshold() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Float64, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float64, Device::cpu(), false);
        grad.data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[1.0, 2.0]);
        p.set_grad(Some(grad));

        let before = p.grad().unwrap().data().as_f64_slice().unwrap().to_vec();
        let mut params = vec![&mut p];
        let norm = GradientUtils::clip_grad_norm(&mut params, 10.0).unwrap();
        assert!((norm - (5.0f64).sqrt()).abs() < 1e-10);

        let after = params[0].grad().unwrap().data().as_f64_slice().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn test_clip_grad_value_for_float_and_non_float() {
        let mut f32_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let mut f64_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Float64, Device::cpu(), true);
        let mut i32_param =
            Tensor::zeros(Shape::new(vec![3]), DataType::Int32, Device::cpu(), true);

        let mut grad_f32 =
            Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), false);
        grad_f32
            .data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[-3.0, 0.5, 9.0]);
        let mut grad_f64 =
            Tensor::ones(Shape::new(vec![3]), DataType::Float64, Device::cpu(), false);
        grad_f64
            .data_mut()
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[-2.0, 0.1, 8.0]);
        let mut grad_i32 = Tensor::ones(Shape::new(vec![3]), DataType::Int32, Device::cpu(), false);
        grad_i32
            .data_mut()
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&[-2, 5, 10]);

        f32_param.set_grad(Some(grad_f32));
        f64_param.set_grad(Some(grad_f64));
        i32_param.set_grad(Some(grad_i32));

        let mut params = vec![&mut f32_param, &mut f64_param, &mut i32_param];
        GradientUtils::clip_grad_value(&mut params, -1.0, 1.0).unwrap();

        let f32_data = params[0].grad().unwrap().data().as_f32_slice().unwrap();
        assert_eq!(f32_data, &[-1.0, 0.5, 1.0]);

        let f64_data = params[1].grad().unwrap().data().as_f64_slice().unwrap();
        assert_eq!(f64_data, &[-1.0, 0.1, 1.0]);

        let i32_data = params[2].grad().unwrap().data().as_i32_slice().unwrap();
        assert_eq!(i32_data, &[-2, 5, 10]);
    }

    #[test]
    fn test_scheduler_utils_factory_methods() {
        let linear = SchedulerUtils::linear_warmup(4);
        assert_eq!(linear.get_lr(2, 0.4), 0.2);

        let poly = SchedulerUtils::polynomial_decay(4, 0.1, 1.0);
        assert_eq!(poly.get_lr(4, 0.5), 0.1);

        let multistep = SchedulerUtils::multi_step(vec![4, 2], 0.1);
        assert!((multistep.get_lr(4, 1.0) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_composite_scheduler_default_and_switching() {
        let base_lr = 1.0;
        let scheduler = CompositeScheduler::default()
            .add_scheduler(Box::new(LinearWarmupScheduler::new(2)), 0)
            .add_scheduler(Box::new(MultiStepScheduler::new(vec![1], 0.1)), 5);

        assert_eq!(scheduler.get_lr(0, base_lr), 0.0);
        assert_eq!(scheduler.get_lr(2, base_lr), 1.0);
        assert_eq!(scheduler.get_lr(5, base_lr), 1.0);
        assert!((scheduler.get_lr(6, base_lr) - 0.1).abs() < 1e-12);

        let empty = CompositeScheduler::new();
        assert_eq!(empty.get_lr(10, base_lr), base_lr);
    }

    #[test]
    fn test_clip_grad_norm_zero_max_norm_zeros_float_grads() {
        let mut p = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let mut grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        grad.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        p.set_grad(Some(grad));

        let mut params = vec![&mut p];
        let norm = GradientUtils::clip_grad_norm(&mut params, 0.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-10);

        let scaled = params[0].grad().unwrap().data().as_f32_slice().unwrap();
        assert!(scaled[0].abs() < 1e-6);
        assert!(scaled[1].abs() < 1e-6);
    }

    #[test]
    fn test_composite_scheduler_step_before_first_scheduler_keeps_base_lr() {
        let scheduler = CompositeScheduler::new()
            .add_scheduler(Box::new(LinearWarmupScheduler::new(3)), 5)
            .add_scheduler(Box::new(MultiStepScheduler::new(vec![1], 0.1)), 10);

        assert_eq!(scheduler.get_lr(0, 0.3), 0.3);
        assert_eq!(scheduler.get_lr(4, 0.3), 0.3);
    }
}
