// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "reduction/argminmax.rs"]
mod argminmax_impl;
#[path = "reduction/boolean.rs"]
mod boolean_impl;
#[path = "reduction/core.rs"]
mod core_impl;
#[path = "reduction/logsumexp.rs"]
mod logsumexp_impl;
#[path = "reduction/minmax_indices.rs"]
mod minmax_indices_impl;
#[path = "reduction/nan_minmax.rs"]
mod nan_minmax_impl;
#[path = "reduction/nanquantile.rs"]
mod nanquantile_impl;
#[path = "reduction/quantile.rs"]
mod quantile_impl;
#[path = "reduction/sort.rs"]
mod sort_impl;
#[path = "reduction/sum_prod.rs"]
mod sum_prod_impl;

pub(crate) use self::argminmax_impl::*;
pub use self::boolean_impl::*;
pub use self::core_impl::*;
pub use self::logsumexp_impl::*;
pub(crate) use self::minmax_indices_impl::*;
pub(crate) use self::nan_minmax_impl::*;
pub use self::nanquantile_impl::*;
pub(crate) use self::quantile_impl::*;
pub use self::sort_impl::*;
pub use self::sum_prod_impl::*;
