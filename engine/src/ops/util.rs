// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Small helpers shared across op clusters and their gradient functions.
//!
//! Everything here was previously defined more than once — the same body
//! copied into a forward kernel and again into the matching backward kernel,
//! where the two copies could drift apart without anything noticing.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

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

/// Sigmoid evaluated through whichever of `e^-x` / `e^x` cannot overflow, so a
/// large-magnitude input saturates to 1 or 0 instead of producing `inf/inf`
/// (NaN). Used by the sigmoid/SiLU forward kernels and their gradients.
#[inline]
pub(crate) fn stable_sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// [`stable_sigmoid_f32`] in double precision.
#[inline]
pub(crate) fn stable_sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// Map a linear index into an output tensor onto the corresponding element of
/// a mask that broadcasts against it.
///
/// Used by the masked softmax / log-softmax kernels and their gradients, which
/// must agree exactly on which mask element governs which output position.
pub(crate) fn broadcast_mask_index(
    linear_idx: usize,
    output_dims: &[usize],
    output_strides: &[usize],
    mask_dims: &[usize],
    mask_strides: &[usize],
) -> usize {
    if mask_dims.is_empty() {
        return 0;
    }

    let output_ndim = output_dims.len();
    let mask_ndim = mask_dims.len();
    let mut mask_index = 0usize;

    for i in 0..mask_ndim {
        let output_dim_idx = output_ndim - 1 - i;
        let mask_dim_idx = mask_ndim - 1 - i;
        let stride = output_strides[output_dim_idx];
        let coord = linear_idx
            .checked_div(stride)
            .map_or(0, |quotient| quotient % output_dims[output_dim_idx]);
        let mask_dim = mask_dims[mask_dim_idx];
        let mask_coord = if mask_dim == 1 { 0 } else { coord };
        mask_index += mask_coord * mask_strides[mask_dim_idx];
    }

    mask_index
}

/// A one-element float tensor holding `value`, for the scalar coefficients the
/// loss and gradient kernels multiply through (`1/n`, `2/n`, …).
///
/// Shape is `[1]` rather than `[]` so it broadcasts against any operand.
pub(crate) fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from scalar")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from scalar")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensors only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
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

    #[test]
    fn stable_sigmoid_saturates_instead_of_producing_nan() {
        // The naive `1/(1 + e^-x)` overflows to `inf` for x <= -104 in f32 and
        // then divides inf by inf; these must be plain 0 and 1.
        assert_eq!(stable_sigmoid_f32(-200.0), 0.0);
        assert_eq!(stable_sigmoid_f32(200.0), 1.0);
        assert_eq!(stable_sigmoid_f64(-800.0), 0.0);
        assert_eq!(stable_sigmoid_f64(800.0), 1.0);
        assert_eq!(stable_sigmoid_f32(0.0), 0.5);
        assert_eq!(stable_sigmoid_f64(0.0), 0.5);
        // Symmetry: sigmoid(-x) == 1 - sigmoid(x).
        for x in [0.5f64, 1.0, 3.25, 12.0] {
            assert!((stable_sigmoid_f64(-x) - (1.0 - stable_sigmoid_f64(x))).abs() < 1e-15);
        }
    }

    #[test]
    fn broadcast_mask_index_maps_output_positions_onto_a_broadcast_mask() {
        // output [2, 3] with a [1, 3] mask: both rows read the same mask row.
        let out_dims = [2usize, 3];
        let out_strides = [3usize, 1];
        let mask_dims = [1usize, 3];
        let mask_strides = [3usize, 1];
        let got: Vec<usize> = (0..6)
            .map(|i| broadcast_mask_index(i, &out_dims, &out_strides, &mask_dims, &mask_strides))
            .collect();
        assert_eq!(got, vec![0, 1, 2, 0, 1, 2]);

        // A rank-0 mask always selects its single element.
        assert_eq!(
            broadcast_mask_index(5, &out_dims, &out_strides, &[], &[]),
            0
        );
    }

    #[test]
    fn create_scalar_tensor_builds_a_broadcastable_one_element_tensor() {
        let t = create_scalar_tensor(0.25, DataType::Float32, Device::cpu()).unwrap();
        assert_eq!(t.shape().dims(), &[1]);
        assert_eq!(t.data().as_f32_slice().unwrap(), &[0.25]);
        assert!(!t.requires_grad());

        let t = create_scalar_tensor(0.25, DataType::Float64, Device::cpu()).unwrap();
        assert_eq!(t.data().as_f64_slice().unwrap(), &[0.25]);

        assert!(create_scalar_tensor(1.0, DataType::Int64, Device::cpu()).is_err());
    }
}
