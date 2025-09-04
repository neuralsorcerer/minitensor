// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod adam;
pub mod optimizer;
pub mod rmsprop;
pub mod sgd;
pub mod utils;

#[cfg(test)]
mod tests;

pub use adam::Adam;
pub use optimizer::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, GradientClipping, LearningRateScheduler,
    Optimizer, ParameterGroup, StepLR,
};
pub use rmsprop::RMSprop;
pub use sgd::SGD;
pub use utils::{
    CompositeScheduler, GradientUtils, LinearWarmupScheduler, MultiStepScheduler,
    PolynomialDecayScheduler, SchedulerUtils,
};
