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
/// Gradient function for [`crate::ops::linalg::linear`].
///
/// Both gradients are GEMMs over the operands the forward already had, with an
/// operand transposed by stride rather than by copy:
///
/// ```text
/// grad_input  = grad @ weight       weight is [out, in], so this is plain
/// grad_weight = grad^T @ input      grad is [rows, out], read as [out, rows]
/// ```
pub struct LinearBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub input_ids: [TensorId; 2],
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
}

impl GradientFunction for LinearBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        use crate::ops::linalg::{gemm_f32, gemm_f64, gemm_tn_f32, gemm_tn_f64};
        use crate::tensor::{DataType, Shape, TensorData};
        use std::sync::Arc;

        let mut gradients = FxHashMap::default();
        let weight_dims = self.weight.shape().dims();
        let (out_features, in_features) = (weight_dims[0], weight_dims[1]);
        let rows: usize = self.input.shape().dims()[..self.input.ndim() - 1]
            .iter()
            .product();
        let empty = rows == 0 || in_features == 0 || out_features == 0;

        if self.input_requires_grad {
            let mut data = TensorData::zeros_on_device(
                rows * in_features,
                grad_output.dtype(),
                grad_output.device(),
            );
            if !empty {
                macro_rules! run {
                    ($accessor:ident, $mut_accessor:ident, $gemm:path) => {{
                        let g = grad_output.data().$accessor().ok_or_else(|| {
                            MinitensorError::internal_error("linear backward: grad dtype")
                        })?;
                        let w = self.weight.data().$accessor().ok_or_else(|| {
                            MinitensorError::internal_error("linear backward: weight dtype")
                        })?;
                        let out = data.$mut_accessor().unwrap();
                        unsafe {
                            $gemm(
                                rows,
                                out_features,
                                in_features,
                                g.as_ptr(),
                                w.as_ptr(),
                                out.as_mut_ptr(),
                            )
                        };
                    }};
                }
                match grad_output.dtype() {
                    DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut, gemm_f32),
                    DataType::Float64 => run!(as_f64_slice, as_f64_slice_mut, gemm_f64),
                    _ => {
                        return Err(MinitensorError::invalid_operation(
                            "linear backward only supports floating point tensors",
                        ));
                    }
                }
            }
            let grad_input = Tensor::new(
                Arc::new(data),
                self.input.shape().clone(),
                grad_output.dtype(),
                grad_output.device(),
                false,
            );
            accumulate_grad(&mut gradients, self.input_ids[0], grad_input)?;
        }

        if self.weight_requires_grad {
            let mut data = TensorData::zeros_on_device(
                out_features * in_features,
                grad_output.dtype(),
                grad_output.device(),
            );
            if !empty {
                macro_rules! run {
                    ($accessor:ident, $mut_accessor:ident, $gemm:path) => {{
                        let g = grad_output.data().$accessor().ok_or_else(|| {
                            MinitensorError::internal_error("linear backward: grad dtype")
                        })?;
                        let x = self.input.data().$accessor().ok_or_else(|| {
                            MinitensorError::internal_error("linear backward: input dtype")
                        })?;
                        let out = data.$mut_accessor().unwrap();
                        // `grad` holds the logical `(out, rows)` operand as
                        // `(rows, out)`.
                        unsafe {
                            $gemm(
                                out_features,
                                rows,
                                in_features,
                                g.as_ptr(),
                                x.as_ptr(),
                                out.as_mut_ptr(),
                            )
                        };
                    }};
                }
                match grad_output.dtype() {
                    DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut, gemm_tn_f32),
                    DataType::Float64 => run!(as_f64_slice, as_f64_slice_mut, gemm_tn_f64),
                    _ => {
                        return Err(MinitensorError::invalid_operation(
                            "linear backward only supports floating point tensors",
                        ));
                    }
                }
            }
            let grad_weight = Tensor::new(
                Arc::new(data),
                Shape::new(vec![out_features, in_features]),
                grad_output.dtype(),
                grad_output.device(),
                false,
            );
            accumulate_grad(&mut gradients, self.input_ids[1], grad_weight)?;
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
/// Gradient of [`crate::ops::linalg::det`].
///
/// `d det(A) / dA = det(A) * A^-T` -- Jacobi's formula. The inverse is
/// obtained by solving against the identity rather than by a second
/// factorisation, which is also how `inv` itself is written, so there is one
/// place where a matrix gets inverted and one place that can be wrong about it.
pub struct DetBackward {
    pub input: Tensor,
    /// The forward result, kept so the gradient does not factorise again.
    pub determinant: Tensor,
    pub input_id: TensorId,
    pub ids: [TensorId; 1],
}

/// The transpose of the inverse, which both determinant gradients are built
/// on. `A^-T` is `(A^T)^-1`, so this transposes first and inverts once.
fn inverse_transpose(input: &Tensor) -> Result<Tensor> {
    let transposed = crate::ops::linalg::transpose(
        input,
        (input.ndim() - 2) as isize,
        (input.ndim() - 1) as isize,
    )?;
    crate::ops::linalg::inv(&transposed)
}

/// Give a batch-shaped scalar the two trailing singleton axes that let it
/// broadcast against `[..., n, n]`. A `det` of a single matrix is 0-d, and a
/// batched one is `[b]`; the gradient is a matrix either way.
fn as_matrix_scalar(value: &Tensor) -> Result<Tensor> {
    let mut out = value.clone();
    for _ in 0..2 {
        let axis = out.ndim() as isize;
        out = crate::ops::shape_ops::unsqueeze(&out, axis)?;
    }
    Ok(out)
}

impl GradientFunction for DetBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let inv_t = inverse_transpose(&self.input)?;
        let scale = crate::ops::arithmetic::mul(
            &as_matrix_scalar(grad_output)?,
            &as_matrix_scalar(&self.determinant)?,
        )?;
        let grad = crate::ops::arithmetic::mul(&inv_t, &scale)?;
        accumulate_grad(&mut gradients, self.input_id, grad)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

/// Gradient of the `logabsdet` half of [`crate::ops::linalg::slogdet`].
///
/// `d log|det(A)| / dA = A^-T`, with no determinant factor -- which is the
/// whole reason `slogdet` is the numerically useful form: the gradient stays
/// finite where `det` itself has already overflowed.
///
/// The sign has no gradient. It is locally constant wherever it is defined and
/// undefined exactly where the matrix is singular, so there is nothing to
/// propagate through it.
pub struct SlogdetBackward {
    pub input: Tensor,
    pub input_id: TensorId,
    pub ids: [TensorId; 1],
}

impl GradientFunction for SlogdetBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let inv_t = inverse_transpose(&self.input)?;
        let grad = crate::ops::arithmetic::mul(&inv_t, &as_matrix_scalar(grad_output)?)?;
        accumulate_grad(&mut gradients, self.input_id, grad)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

/// Gradient of [`crate::ops::linalg::cholesky`].
///
/// The factor is the only thing the gradient needs -- not the matrix it came
/// from -- which is the useful property of this factorisation: `A` is
/// recoverable from `L` and the reverse pass never has to hold both.
///
/// The arithmetic itself lives next to the forward, in
/// [`crate::ops::linalg::cholesky_backward_batched`], because it is made of the
/// same two triangular substitutions and the two would drift apart if they were
/// written in different files.
pub struct CholeskyBackward {
    /// The lower factor `L`, detached. `upper = true` composes a transpose on
    /// top of this node, so what arrives here is always the lower one.
    pub factor: Tensor,
    pub input_id: TensorId,
    pub ids: [TensorId; 1],
}

impl GradientFunction for CholeskyBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        use crate::ops::linalg::{cholesky_backward_batched, square_extent};
        use crate::tensor::{DataType, TensorData};
        use std::sync::Arc;

        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = self.factor.shape().clone();
        let (batch, n) = square_extent(&shape);
        // The incoming gradient is whatever the graph produced; the kernel walks
        // it row-major alongside the factor.
        let grad = grad_output.contiguous()?;
        let mut data = TensorData::zeros_on_device(shape.numel(), grad.dtype(), grad.device());

        macro_rules! run {
            ($accessor:ident, $accessor_mut:ident) => {{
                let factor = self.factor.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("cholesky backward: factor dtype")
                })?;
                let upstream = grad.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("cholesky backward: grad dtype")
                })?;
                let out = data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("cholesky backward: output dtype")
                })?;
                cholesky_backward_batched(factor, upstream, out, n, batch);
            }};
        }

        match grad.dtype() {
            DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut),
            DataType::Float64 => run!(as_f64_slice, as_f64_slice_mut),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "cholesky backward only supports floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(Arc::new(data), shape, grad.dtype(), grad.device(), false);
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

/// Gradient of [`crate::ops::linalg::qr`].
///
/// The factorisation has two outputs and the graph gives a node one gradient at
/// a time, so `Q` and `R` get one of these each and the engine adds what they
/// produce. That is not an approximation: the gradient is linear in the pair, so
/// `grad(gQ, gR) = grad(gQ, 0) + grad(0, gR)` exactly, and each node computes
/// its half with the other term simply absent rather than multiplied by zeros.
///
/// The formula, for `A` at least as tall as it is wide:
///
/// ```text
/// M  = R gR^T - gQ^T Q
/// gA = [gQ + Q copyltu(M)] R^-T
/// ```
///
/// where `copyltu` mirrors the lower triangle into the upper. The mirroring is
/// what encodes the constraint that `Q^T Q` stays the identity: only the
/// symmetric part of the perturbation is realisable, and the antisymmetric part
/// would move `Q` off the orthogonal matrices where the factorisation has no
/// meaning.
///
/// A wide `A` is handled by splitting it at the square block. `Q` is fixed by
/// the first `m` columns alone, so `R2 = Q^T A2` is an ordinary product whose
/// gradient adds into `Q`'s before the square case runs on the rest.
pub struct QrBackward {
    pub input: Tensor,
    pub q: Tensor,
    pub r: Tensor,
    /// Which of the two outputs this node stands for.
    pub from_q: bool,
    pub input_id: TensorId,
    pub ids: [TensorId; 1],
}

/// A tensor of zeros shaped like `tensor`, carrying no gradient of its own.
fn zeros_like(tensor: &Tensor) -> Tensor {
    Tensor::zeros(
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    )
}

/// `transpose` on the last two axes, which is all this file ever wants.
fn swap_last_two(tensor: &Tensor) -> Result<Tensor> {
    crate::ops::linalg::transpose(
        tensor,
        (tensor.ndim() - 2) as isize,
        (tensor.ndim() - 1) as isize,
    )
}

/// The lower triangle of `m`, mirrored into the upper.
///
/// `tril(M) + tril(M, -1)^T` -- the diagonal is taken once, everything below it
/// twice, and nothing above it is read at all.
fn copy_lower_to_upper(m: &Tensor) -> Result<Tensor> {
    let lower = crate::ops::linalg::tril(m, 0)?;
    let strict = crate::ops::linalg::tril(m, -1)?;
    crate::ops::arithmetic::add(&lower, &swap_last_two(&strict)?)
}

/// The gradient for a `Q R` whose `R` is square, given whichever of the two
/// output gradients is present.
fn qr_grad_square(
    q: &Tensor,
    r: &Tensor,
    grad_q: Option<&Tensor>,
    grad_r: Option<&Tensor>,
) -> Result<Tensor> {
    // M = R gR^T - gQ^T Q, with an absent term left out rather than zeroed.
    let mut m = match grad_r {
        Some(gr) => Some(crate::ops::linalg::matmul(r, &swap_last_two(gr)?)?),
        None => None,
    };
    if let Some(gq) = grad_q {
        let term = crate::ops::linalg::matmul(&swap_last_two(gq)?, q)?;
        m = Some(match m {
            Some(existing) => crate::ops::arithmetic::sub(&existing, &term)?,
            None => crate::ops::arithmetic::neg(&term)?,
        });
    }

    let mut inner = match m {
        Some(m) => Some(crate::ops::linalg::matmul(q, &copy_lower_to_upper(&m)?)?),
        None => None,
    };
    if let Some(gq) = grad_q {
        inner = Some(match inner {
            Some(existing) => crate::ops::arithmetic::add(&existing, gq)?,
            None => gq.clone(),
        });
    }
    let inner = match inner {
        Some(inner) => inner,
        // Nothing reached this node, so nothing leaves it.
        None => return Ok(zeros_like(q)),
    };

    // gA = inner R^-T, which is R gA^T = inner^T -- one solve rather than an
    // inverse, and the one place a singular `R` is reported.
    let solved = crate::ops::linalg::solve(r, &swap_last_two(&inner)?)?;
    swap_last_two(&solved)
}

impl GradientFunction for QrBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let dims = self.input.shape().dims();
        let (rows, cols) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        let (grad_q, grad_r) = if self.from_q {
            (Some(grad_output), None)
        } else {
            (None, Some(grad_output))
        };

        let grad_input = if rows >= cols {
            qr_grad_square(&self.q, &self.r, grad_q, grad_r)?
        } else {
            // A = [A1 | A2] with A1 square. `Q` comes from A1 alone, and
            // `R2 = Q^T A2` contributes to both halves.
            let last = dims.len() - 1;
            let r1 = crate::ops::shape_ops::slice(&self.r, last as isize, 0, rows, 1)?;
            let r2 = crate::ops::shape_ops::slice(&self.r, last as isize, rows, cols, 1)?;
            let a2 = crate::ops::shape_ops::slice(&self.input, last as isize, rows, cols, 1)?;
            let _ = r2;

            let (grad_r1, grad_r2) = match grad_r {
                Some(gr) => (
                    Some(crate::ops::shape_ops::slice(gr, last as isize, 0, rows, 1)?),
                    Some(crate::ops::shape_ops::slice(
                        gr,
                        last as isize,
                        rows,
                        cols,
                        1,
                    )?),
                ),
                None => (None, None),
            };

            // R2 = Q^T A2 sends gradient to both operands.
            let effective_q = match &grad_r2 {
                Some(gr2) => {
                    let term = crate::ops::linalg::matmul(&a2, &swap_last_two(gr2)?)?;
                    Some(match grad_q {
                        Some(gq) => crate::ops::arithmetic::add(gq, &term)?,
                        None => term,
                    })
                }
                None => grad_q.cloned(),
            };

            let grad_a1 = qr_grad_square(&self.q, &r1, effective_q.as_ref(), grad_r1.as_ref())?;
            let grad_a2 = match &grad_r2 {
                Some(gr2) => crate::ops::linalg::matmul(&self.q, gr2)?,
                None => zeros_like(&a2),
            };
            crate::ops::shape_ops::concatenate(&[&grad_a1, &grad_a2], last as isize)?
        };

        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

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
