// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "activation/elementwise.rs"]
mod elementwise_impl;
#[path = "activation/hyperbolic.rs"]
mod hyperbolic_impl;
#[path = "activation/power.rs"]
mod power_impl;
#[path = "activation/softmax.rs"]
mod softmax_impl;
#[path = "activation/trigonometry.rs"]
mod trigonometry_impl;
#[path = "activation/units.rs"]
pub mod units;

pub use self::elementwise_impl::*;
pub use self::hyperbolic_impl::*;
pub use self::power_impl::*;
pub(crate) use self::softmax_impl::*;
pub use self::trigonometry_impl::*;
pub use self::units::*;

// Shared tests for the modules above. Previously `activation/advanced.rs`,
// where six hundred lines of tests for the other five modules sat around a
// single implementation (`ceil_f64`) that belonged with its siblings in
// `power`; the name promised an "advanced" set of operations that was never
// there.
#[cfg(test)]
#[path = "activation/tests.rs"]
mod tests;
