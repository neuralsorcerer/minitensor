// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod lion;
pub mod nadam;
pub mod optimizer;
pub mod radam;
pub mod rmsprop;
pub mod rprop;
pub mod sgd;
pub mod utils;

#[cfg(test)]
mod tests;

pub use adadelta::Adadelta;
pub use adagrad::Adagrad;
pub use adam::{Adam, AdamW};
pub use adamax::Adamax;
pub use lion::Lion;
pub use nadam::NAdam;
pub use optimizer::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, GradientClipping, LearningRateScheduler,
    Optimizer, ParamGroups, ParameterGroup, StepLR, check_param_grad_match, parameter_gradient,
};
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use rprop::Rprop;
pub use sgd::SGD;
pub use utils::{
    CompositeScheduler, GradientUtils, LinearWarmupScheduler, MultiStepScheduler,
    PolynomialDecayScheduler, SchedulerUtils,
};
