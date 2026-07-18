// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{MinitensorError, Result};

/// Resolve a possibly negative dimension index against `ndim`, erroring when
/// it falls outside `[-ndim, ndim)`. Shared by the shape, linalg, and
/// reduction clusters.
pub(crate) fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let dim = if dim < 0 { dim + ndim as isize } else { dim };
    if dim < 0 || dim >= ndim as isize {
        Err(MinitensorError::index_error(dim, 0, ndim))
    } else {
        Ok(dim as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_dims_wrap_and_bounds_error() {
        assert_eq!(normalize_dim(-1, 3).unwrap(), 2);
        assert_eq!(normalize_dim(0, 3).unwrap(), 0);
        assert_eq!(normalize_dim(2, 3).unwrap(), 2);
        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
        assert!(normalize_dim(0, 0).is_err());
    }
}
