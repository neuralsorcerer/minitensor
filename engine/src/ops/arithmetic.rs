// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Element-wise arithmetic ops. The dtype-specialized kernel bodies these
//! delegate to live in [`crate::ops::kernels`].

#[path = "arithmetic/close.rs"]
pub(crate) mod close;
#[path = "arithmetic/difference_of_products.rs"]
mod difference_of_products;
#[path = "arithmetic/elementwise.rs"]
mod elementwise_impl;
#[path = "arithmetic/evenly_spaced.rs"]
mod evenly_spaced;
#[path = "arithmetic/exact_sum.rs"]
pub(crate) mod exact_sum;
#[path = "arithmetic/rounded_once.rs"]
pub(crate) mod rounded_once;

pub use self::difference_of_products::difference_of_products;
pub use self::elementwise_impl::*;
pub use self::evenly_spaced::{
    evenly_spaced_data, evenly_spaced_point, stepped_data, stepped_point,
};
