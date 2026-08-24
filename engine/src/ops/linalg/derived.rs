// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The operations that are a factorisation read out a particular way.
//!
//! `svd` answers how a matrix stretches space, and four of the questions people
//! actually ask are that answer with one line of arithmetic on it: the
//! pseudo-inverse inverts the directions that are invertible, the rank counts
//! the directions that survive rounding, the condition number is the ratio of
//! the largest to the smallest, and least squares is the pseudo-inverse applied
//! to a right-hand side. None of them is hard once the factorisation exists and
//! none of them is *possible* without it, which is why they live together and
//! why they arrived with it.
//!
//! What they share is the tolerance. A singular value is never exactly zero in
//! floating point, so every one of these has to decide which of them count, and
//! the answer is the same everywhere: anything at or below `max(m, n)` roundings
//! of the largest singular value is indistinguishable from zero, because that is
//! the accuracy the factorisation itself offers. One [`default_tolerance`], used
//! by all of them, rather than four thresholds that could drift apart.
//!
//! [`matrix_power`] is here for a different reason -- it is `inv` and `matmul`
//! rather than `svd` -- but it is the same kind of thing: a name for something a
//! caller would otherwise write out and get subtly wrong, in its case by
//! multiplying `n` times instead of `log n` and by forgetting that a negative
//! power has to invert first.

use crate::{
    error::{MinitensorError, Result},
    ops::{arithmetic, linalg, selection, shape_ops},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// The batch shape and the two matrix extents, rejecting what cannot factor.
fn layout(tensor: &Tensor, op: &str) -> Result<(Vec<usize>, usize, usize)> {
    linalg::matrix_layout(tensor, op)
}

/// A one-element tensor holding `value`, for the elementwise ops to broadcast.
fn scalar(value: f64, like: &Tensor) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, like.dtype(), like.device());
    match like.dtype() {
        DataType::Float32 => {
            let slice = data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("failed to build a constant"))?;
            slice[0] = value as f32;
        }
        _ => {
            let slice = data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("failed to build a constant"))?;
            slice[0] = value;
        }
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        like.dtype(),
        like.device(),
        false,
    ))
}

/// How small a singular value has to be before it counts as zero.
///
/// `max(m, n) * eps`, relative to the largest singular value, which is LAPACK's
/// convention and PyTorch's default. It is not arbitrary: the factorisation
/// guarantees each value to within roughly `eps` times the largest, and a
/// perturbation of the matrix by a single rounding can move a value by that
/// much times the dimension. A value below it is a value the input did not
/// determine.
fn default_tolerance(dtype: DataType, m: usize, n: usize) -> f64 {
    let epsilon = match dtype {
        DataType::Float32 => f32::EPSILON as f64,
        _ => f64::EPSILON,
    };
    m.max(n) as f64 * epsilon
}

/// The largest singular value of each matrix, shaped to broadcast against the
/// whole stack of them.
fn largest(values: &Tensor) -> Result<Tensor> {
    values.max(Some(values.ndim() as isize - 1), true)
}

/// `1 / s` where `s` counts, and zero where it does not.
///
/// The zeros are the whole point rather than a guard against dividing: the
/// pseudo-inverse of a singular matrix inverts what it can and sends the rest to
/// nothing, and that is what makes it the least-squares solution rather than a
/// failure. `where` picks before the division is used, so no infinity is formed
/// and none has to be cleaned up afterwards.
fn invert_above(values: &Tensor, cutoff: &Tensor) -> Result<Tensor> {
    let keep = comparison_gt(values, cutoff)?;
    let one = scalar(1.0, values)?;
    let safe = arithmetic::add(values, &comparison_as_zero_guard(&keep, values)?)?;
    let inverted = arithmetic::div(&one, &safe)?;
    let zero = scalar(0.0, values)?;
    selection::where_op(&keep, &inverted, &zero)
}

/// `values > cutoff`, broadcast.
fn comparison_gt(values: &Tensor, cutoff: &Tensor) -> Result<Tensor> {
    crate::ops::comparison::gt(values, cutoff)
}

/// One where `keep` is false and zero where it is true, in `values`' dtype.
///
/// Added to the values before the reciprocal so that a discarded value -- which
/// may be exactly zero -- is divided as one rather than producing an infinity
/// that `where` would then have to discard. The result is thrown away either
/// way; this only keeps a NaN from ever existing.
fn comparison_as_zero_guard(keep: &Tensor, values: &Tensor) -> Result<Tensor> {
    let one = scalar(1.0, values)?;
    let zero = scalar(0.0, values)?;
    selection::where_op(keep, &zero, &one)
}

/// The Moore-Penrose pseudo-inverse of every matrix in a stack.
///
/// `V diag(1/s) U^T` over the singular values that count, which is the unique
/// matrix satisfying the four Penrose conditions and, for a system with no exact
/// solution, the one that returns the least-squares answer of smallest norm.
/// For an invertible square matrix it is the inverse, computed a more expensive
/// and more careful way.
///
/// `rcond` is relative to the largest singular value; `None` takes
/// [`default_tolerance`]. Singular values at or below it are treated as zero and
/// their directions are dropped, which is what makes this defined for a
/// rank-deficient or non-square matrix at all.
pub fn pinv(tensor: &Tensor, rcond: Option<f64>) -> Result<Tensor> {
    let (_, m, n) = layout(tensor, "pinv")?;
    let (u, s, vt) = linalg::svd(tensor, false)?;

    let relative = rcond.unwrap_or_else(|| default_tolerance(tensor.dtype(), m, n));
    let cutoff = arithmetic::mul(&largest(&s)?, &scalar(relative, &s)?)?;
    let inverted = invert_above(&s, &cutoff)?;

    // V diag(1/s) U^T, with the diagonal applied by broadcasting across the
    // columns of `V` rather than by forming a matrix of mostly zeros.
    let v = linalg::transpose(&vt, -2, -1)?;
    let scaled = arithmetic::mul(
        &v,
        &shape_ops::unsqueeze(&inverted, vt.ndim() as isize - 2)?,
    )?;
    linalg::matmul(&scaled, &linalg::transpose(&u, -2, -1)?)
}

/// How many singular values of each matrix are distinguishable from zero.
///
/// The only numerically meaningful definition of rank for inexact entries: a
/// matrix that is one rounding away from rank three has rank three, whatever
/// exact arithmetic on its stored digits would say. Returns `int64`, one number
/// per matrix in the stack.
///
/// `tol` is absolute when given, and relative to the largest singular value
/// through [`default_tolerance`] when it is not.
pub fn matrix_rank(tensor: &Tensor, tol: Option<f64>) -> Result<Tensor> {
    let (_, m, n) = layout(tensor, "matrix_rank")?;
    let values = linalg::svdvals(&tensor.detach())?;

    let cutoff = match tol {
        Some(absolute) => scalar(absolute, &values)?,
        None => arithmetic::mul(
            &largest(&values)?,
            &scalar(default_tolerance(tensor.dtype(), m, n), &values)?,
        )?,
    };
    let keep = comparison_gt(&values, &cutoff)?;
    let axis = values.ndim() as isize - 1;
    keep.astype(DataType::Int64)?.sum(Some(vec![axis]), false)
}

/// The 2-norm condition number: the ratio of the largest singular value to the
/// smallest.
///
/// How much a relative error in the input can be amplified in the output of a
/// solve. A condition number near `1 / eps` means the answer has no correct
/// digits left, and infinity means the matrix is singular -- which is reported
/// as infinity rather than as an error, because that is the true answer and a
/// caller comparing against a threshold should not have to catch it.
pub fn cond(tensor: &Tensor) -> Result<Tensor> {
    let values = linalg::svdvals(&tensor.detach())?;
    let axis = Some(values.ndim() as isize - 1);
    let biggest = values.max(axis, false)?;
    let smallest = values.min(axis, false)?;
    arithmetic::div(&biggest, &smallest)
}

/// The least-squares solution of `A x = b`, for any `A` at all.
///
/// Minimises `||A x - b||`, and among the solutions that do -- there are many
/// when `A` is rank deficient -- returns the one of smallest norm. `solve`
/// needs a square non-singular `A` and `qr` needs full column rank; this needs
/// neither, which is the whole reason it exists.
///
/// `b` may be a matrix of right-hand sides or a single vector, and the result
/// matches: a vector in gives a vector out.
pub fn lstsq(a: &Tensor, b: &Tensor, rcond: Option<f64>) -> Result<Tensor> {
    let (_, m, _) = layout(a, "lstsq")?;

    // A single right-hand side is a matrix with one column for the duration.
    let vector = b.ndim() + 1 == a.ndim();
    let rhs = if vector {
        shape_ops::unsqueeze(b, b.ndim() as isize)?
    } else {
        b.clone()
    };
    let rows = rhs.shape().dims()[rhs.ndim() - 2];
    if rows != m {
        return Err(MinitensorError::invalid_operation(format!(
            "lstsq: the right-hand side has {rows} rows but the matrix has {m}"
        )));
    }

    let solution = linalg::matmul(&pinv(a, rcond)?, &rhs)?;
    if vector {
        shape_ops::squeeze(&solution, Some(solution.ndim() as isize - 1))
    } else {
        Ok(solution)
    }
}

/// `A` multiplied by itself `power` times.
///
/// Zero gives the identity, a negative power inverts first, and everything else
/// goes by repeated squaring -- `log2(power)` matrix products rather than
/// `power` of them, which is the difference between instant and hopeless at
/// `power = 1000` and is the reason to have the name at all.
pub fn matrix_power(tensor: &Tensor, power: i64) -> Result<Tensor> {
    let (batch_dims, m, n) = layout(tensor, "matrix_power")?;
    if m != n {
        return Err(MinitensorError::invalid_operation(format!(
            "matrix_power expects square matrices, got {m} by {n}"
        )));
    }

    if power == 0 {
        return identity_like(tensor, &batch_dims, m);
    }
    // The inverse is taken once, before the squaring, rather than inverting the
    // result: they agree in exact arithmetic and inverting a matrix that has
    // already been raised to a large power inverts its condition number too.
    let base = if power < 0 {
        linalg::inv(tensor)?
    } else {
        tensor.clone()
    };

    let mut remaining = power.unsigned_abs();
    let mut square = base;
    let mut result: Option<Tensor> = None;
    loop {
        if remaining & 1 == 1 {
            result = Some(match result {
                Some(acc) => linalg::matmul(&acc, &square)?,
                None => square.clone(),
            });
        }
        remaining >>= 1;
        if remaining == 0 {
            break;
        }
        square = linalg::matmul(&square, &square)?;
    }
    result.ok_or_else(|| MinitensorError::internal_error("matrix_power: no factors accumulated"))
}

/// A stack of identity matrices shaped like `tensor`'s.
fn identity_like(tensor: &Tensor, batch_dims: &[usize], n: usize) -> Result<Tensor> {
    let mut dims = batch_dims.to_vec();
    dims.extend_from_slice(&[n, n]);
    let shape = Shape::new(dims);
    let batch: usize = batch_dims.iter().product();
    let mut data = TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    macro_rules! fill {
        ($accessor:ident, $ty:ty) => {{
            let slice = data.$accessor().ok_or_else(|| {
                MinitensorError::internal_error("matrix_power: failed to build the identity")
            })?;
            for b in 0..batch {
                for i in 0..n {
                    slice[b * n * n + i * n + i] = 1 as $ty;
                }
            }
        }};
    }
    match tensor.dtype() {
        DataType::Float32 => fill!(as_f32_slice_mut, f32),
        DataType::Float64 => fill!(as_f64_slice_mut, f64),
        DataType::Int32 => fill!(as_i32_slice_mut, i32),
        DataType::Int64 => fill!(as_i64_slice_mut, i64),
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "matrix_power is not defined for bool tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        shape,
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}
