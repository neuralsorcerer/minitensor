// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[path = "linalg/cholesky.rs"]
mod cholesky_impl;
#[path = "linalg/derived.rs"]
mod derived_impl;
#[path = "linalg/determinant.rs"]
mod determinant_impl;
#[path = "linalg/diagonal.rs"]
mod diagonal_impl;
#[path = "linalg/eigen.rs"]
mod eigen_impl;
#[path = "linalg/linear.rs"]
mod linear_impl;
#[path = "linalg/lu.rs"]
mod lu_impl;
#[path = "linalg/matmul.rs"]
mod matmul_impl;
#[path = "linalg/qr.rs"]
mod qr_impl;
#[path = "linalg/reflector.rs"]
pub(crate) mod reflector;
#[path = "linalg/rotation.rs"]
pub(crate) mod rotation;
#[path = "linalg/svd.rs"]
mod svd_impl;
#[path = "linalg/triangular.rs"]
mod triangular_impl;

pub use self::cholesky_impl::*;
pub use self::derived_impl::*;
pub use self::determinant_impl::*;
pub use self::diagonal_impl::*;
pub use self::eigen_impl::*;
pub use self::linear_impl::*;
pub use self::lu_impl::*;
pub use self::matmul_impl::*;
pub use self::qr_impl::*;
pub use self::svd_impl::*;
pub(crate) use self::triangular_impl::*;

use crate::error::Result;
use crate::tensor::{DataType, Tensor};

/// A factorisation's float64 results, rounded to float32 once.
///
/// A float32 matrix is exact in float64, so a factorisation carried there and
/// rounded at the end has float32 factors as near the exact ones as float32
/// holds; carried in float32 arithmetic, an `eigh` residual came out twenty to
/// fifty times float32's own, its eigenvectors' orthogonality thirty to
/// seventy, and an SVD's small singular values three orders less accurate. The
/// float64 factorisations are as fast as the float32 ones for `solve`, `inv`,
/// `det` and `cholesky`, and at most 1.7 times slower for `qr`, `eigh` and
/// `svd`. The conversions are in the graph, so a gradient comes back float32.
pub(crate) trait Narrow: Sized {
    fn narrowed(self) -> Result<Self>;
}

impl Narrow for Tensor {
    fn narrowed(self) -> Result<Self> {
        if self.dtype() == DataType::Float64 {
            self.astype(DataType::Float32)
        } else {
            Ok(self)
        }
    }
}

impl<A: Narrow, B: Narrow> Narrow for (A, B) {
    fn narrowed(self) -> Result<Self> {
        Ok((self.0.narrowed()?, self.1.narrowed()?))
    }
}

impl<A: Narrow, B: Narrow, C: Narrow> Narrow for (A, B, C) {
    fn narrowed(self) -> Result<Self> {
        Ok((self.0.narrowed()?, self.1.narrowed()?, self.2.narrowed()?))
    }
}

/// `tensor` in float64, for a float32 factorisation to be carried there; see
/// [`Narrow`].
pub(crate) fn widened(tensor: &Tensor) -> Result<Tensor> {
    tensor.astype(DataType::Float64)
}
