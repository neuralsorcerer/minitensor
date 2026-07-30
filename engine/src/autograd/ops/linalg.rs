// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Gradient functions for the linear-algebra operations in
//! [`crate::ops::linalg`].

use super::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::Tensor,
};
use rustc_hash::FxHashMap;

/// Gradient function for dot product
pub struct DotBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}
impl GradientFunction for DotBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        if self.lhs_requires_grad {
            let grad = crate::ops::arithmetic::mul(&self.rhs, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[0], grad)?;
        }

        if self.rhs_requires_grad {
            let grad = crate::ops::arithmetic::mul(&self.lhs, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for matrix multiplication
pub struct MatMulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}
impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        if self.lhs.ndim() < 2 || self.rhs.ndim() < 2 {
            return Err(MinitensorError::invalid_operation(
                "MatMulBackward requires tensors with at least 2 dimensions",
            ));
        }

        if self.lhs_requires_grad {
            let rhs_t = crate::ops::linalg::transpose(
                &self.rhs,
                (self.rhs.ndim() - 2) as isize,
                (self.rhs.ndim() - 1) as isize,
            )?;
            let lhs_grad = crate::ops::linalg::matmul(grad_output, &rhs_t)?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        if self.rhs_requires_grad {
            let lhs_t = crate::ops::linalg::transpose(
                &self.lhs,
                (self.lhs.ndim() - 2) as isize,
                (self.lhs.ndim() - 1) as isize,
            )?;
            let rhs_grad = crate::ops::linalg::matmul(&lhs_t, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
/// Gradient function for solving linear systems.
pub struct SolveBackward {
    pub lhs: Tensor,
    pub solution: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}
impl GradientFunction for SolveBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        let lhs_t = crate::ops::linalg::transpose(
            &self.lhs,
            (self.lhs.ndim() - 2) as isize,
            (self.lhs.ndim() - 1) as isize,
        )?;

        if self.rhs_requires_grad {
            let grad_rhs = crate::ops::linalg::solve(&lhs_t, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], grad_rhs)?;
        }

        if self.lhs_requires_grad {
            let solution_view = if self.solution.ndim() == self.lhs.ndim() - 1 {
                crate::ops::shape_ops::unsqueeze(&self.solution, self.solution.ndim() as isize)?
            } else {
                self.solution.clone()
            };

            let grad_output_view = if grad_output.ndim() == self.lhs.ndim() - 1 {
                crate::ops::shape_ops::unsqueeze(grad_output, grad_output.ndim() as isize)?
            } else {
                grad_output.clone()
            };

            let solution_t = crate::ops::linalg::transpose(
                &solution_view,
                (solution_view.ndim() - 2) as isize,
                (solution_view.ndim() - 1) as isize,
            )?;
            let gram = crate::ops::linalg::matmul(&grad_output_view, &solution_t)?;
            let lhs_grad = crate::ops::linalg::solve(&lhs_t, &gram)?;
            let lhs_grad = crate::ops::arithmetic::neg(&lhs_grad)?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
