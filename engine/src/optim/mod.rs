// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod optimizer;
pub mod sgd;
pub mod adam;
pub mod rmsprop;
pub mod utils;

#[cfg(test)]
mod tests;

pub use optimizer::{
    Optimizer, ParameterGroup, GradientClipping, LearningRateScheduler,
    ConstantLR, StepLR, ExponentialLR, CosineAnnealingLR
};
pub use sgd::SGD;
pub use adam::Adam;
pub use rmsprop::RMSprop;
pub use utils::{
    GradientUtils, SchedulerUtils,
    LinearWarmupScheduler, PolynomialDecayScheduler, MultiStepScheduler, CompositeScheduler
};