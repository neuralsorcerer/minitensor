// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Determinants and the matrix inverse.
//!
//! These are the three square-matrix operations that cannot be assembled from
//! anything else the library has, and their absence ruled out a whole category
//! of work: a Gaussian log-likelihood needs `slogdet`, whitening and a
//! precision matrix need `inv`, and there is no way to write either out of
//! `matmul` and `solve` alone.
//!
//! [`inv`] *is* written out of `solve`, though -- `A @ X = I` -- which is
//! deliberate. It inherits the pivoting, the batching, the singularity check
//! and the gradient of a routine that was already tested, rather than repeating
//! any of them: the gradient of `A^-1 B` with respect to `A` is
//! `-A^-T G X^T`, and at `B = I` that is exactly the derivative of the inverse.
//!
//! [`det`] and [`slogdet`] do need their own elimination, because the pivot
//! sign and the diagonal of `U` are what the answer is made of and `solve`
//! discards both. `slogdet` exists next to `det` for the reason it does
//! everywhere: the determinant of a large matrix overflows long before it
//! stops being useful -- a 200x200 matrix of standard normals has a
//! determinant around 1e186, and one twice that size has none that a float64
//! can hold -- while its logarithm is an ordinary number.

use crate::ops::linalg::solve;
use crate::{
    autograd::{DetBackward, NoGradGuard, SlogdetBackward, with_grad_fn},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// The batch shape and matrix order of a stack of square matrices.
fn square_layout(tensor: &Tensor, op: &str) -> Result<(Vec<usize>, usize)> {
    let ndim = tensor.ndim();
    if ndim < 2 {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects at least 2 dimensions"
        )));
    }
    let dims = tensor.shape().dims();
    let n = dims[ndim - 1];
    if dims[ndim - 2] != n {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects square matrices"
        )));
    }
    if !matches!(tensor.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} supports only Float32 and Float64 tensors"
        )));
    }
    Ok((dims[..ndim - 2].to_vec(), n))
}

/// Row-reduce one matrix in place, returning `(sign, diagonal of U)`.
///
/// The same partial-pivoting elimination `solve` runs, kept separate because it
/// answers a different question: `solve` throws the factorisation away once the
/// right-hand side has been updated, while the determinant *is* the
/// factorisation -- the product of the pivots, negated once per row swap.
///
/// A singular matrix is not an error here, unlike in `solve`. A determinant of
/// zero is a fact about the matrix and the caller asked for it; a zero pivot is
/// how that fact arrives.
fn lu_pivots<T>(matrix: &mut [T], n: usize) -> (bool, Vec<T>)
where
    T: Copy
        + Default
        + PartialOrd
        + std::ops::SubAssign
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>,
{
    let mut negated = false;
    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_val = abs(matrix[k * n + k]);
        for i in (k + 1)..n {
            let candidate = abs(matrix[i * n + k]);
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = i;
            }
        }

        if pivot_val == T::default() {
            // A zero column below the diagonal: the matrix is singular and the
            // determinant is zero. Returning the zero pivot lets the caller say
            // so without a special case.
            let mut diag = vec![T::default(); n];
            for (i, slot) in diag.iter_mut().enumerate() {
                *slot = matrix[i * n + i];
            }
            return (negated, diag);
        }

        if pivot_row != k {
            for col in 0..n {
                matrix.swap(k * n + col, pivot_row * n + col);
            }
            negated = !negated;
        }

        let pivot = matrix[k * n + k];
        for i in (k + 1)..n {
            let factor = matrix[i * n + k] / pivot;
            matrix[i * n + k] = T::default();
            for j in (k + 1)..n {
                let idx = i * n + j;
                matrix[idx] -= factor * matrix[k * n + j];
            }
        }
    }

    let mut diag = vec![T::default(); n];
    for (i, slot) in diag.iter_mut().enumerate() {
        *slot = matrix[i * n + i];
    }
    (negated, diag)
}

fn abs<T>(value: T) -> T
where
    T: Copy + PartialOrd + std::ops::Neg<Output = T> + Default,
{
    if value < T::default() { -value } else { value }
}

macro_rules! determinant_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        /// `(determinant, sign, log|determinant|)` for every matrix in the
        /// stack. All three come from one factorisation because computing them
        /// separately would factorise the same matrix twice, and because the
        /// two answers must agree: `det` and `slogdet` disagreeing about
        /// whether a matrix is singular would be the worst kind of bug to find.
        fn $name(input: &Tensor, batch: usize, n: usize) -> Result<(Vec<$ty>, Vec<$ty>, Vec<$ty>)> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("determinant: dtype does not match the slice")
            })?;
            let stride = n * n;
            let mut dets = vec![0 as $ty; batch];
            let mut signs = vec![0 as $ty; batch];
            let mut logs = vec![0 as $ty; batch];
            let mut scratch = vec![0 as $ty; stride];

            for b in 0..batch {
                scratch.copy_from_slice(&data[b * stride..(b + 1) * stride]);
                let (negated, diag) = lu_pivots(&mut scratch, n);

                let mut det = if negated { -1.0 } else { 1.0 } as $ty;
                let mut sign = det;
                let mut log_abs = 0.0 as $ty;
                for &d in &diag {
                    det *= d;
                    if d < 0.0 {
                        sign = -sign;
                    }
                    // `ln(0)` is `-inf`, which is the right answer: a singular
                    // matrix has log-determinant negative infinity, and any
                    // model scoring one should see that rather than a number.
                    log_abs += abs(d).ln();
                }
                if !log_abs.is_finite() && log_abs.is_sign_negative() {
                    // Singular: the sign carries no information, and every
                    // implementation reports it as zero rather than +/-1.
                    sign = 0.0;
                }
                dets[b] = det;
                signs[b] = sign;
                logs[b] = log_abs;
            }
            Ok((dets, signs, logs))
        }
    };
}

determinant_kernel!(determinants_f32, f32, as_f32_slice);
determinant_kernel!(determinants_f64, f64, as_f64_slice);

/// The three answers one factorisation produces, kept together in the dtype
/// they were computed in.
///
/// `det` and `slogdet` are the same walk over the same pivots; splitting them
/// into separate passes would factorise each matrix twice and, worse, leave
/// room for the two to disagree about whether a matrix is singular.
enum Determinants {
    F32 {
        det: Vec<f32>,
        sign: Vec<f32>,
        log_abs: Vec<f32>,
    },
    F64 {
        det: Vec<f64>,
        sign: Vec<f64>,
        log_abs: Vec<f64>,
    },
}

fn factorise(tensor: &Tensor, batch: usize, n: usize) -> Result<Determinants> {
    match tensor.dtype() {
        DataType::Float32 => {
            let (det, sign, log_abs) = determinants_f32(tensor, batch, n)?;
            Ok(Determinants::F32 { det, sign, log_abs })
        }
        _ => {
            let (det, sign, log_abs) = determinants_f64(tensor, batch, n)?;
            Ok(Determinants::F64 { det, sign, log_abs })
        }
    }
}

/// Wrap one batch-shaped result vector as a tensor.
fn batch_tensor(
    batch_dims: &[usize],
    dtype: DataType,
    device: crate::device::Device,
    f32_values: Vec<f32>,
    f64_values: Vec<f64>,
    requires_grad: bool,
) -> Tensor {
    let data = match dtype {
        DataType::Float32 => TensorData::from_vec_f32(f32_values, device),
        _ => TensorData::from_vec_f64(f64_values, device),
    };
    Tensor::new(
        Arc::new(data),
        Shape::new(batch_dims.to_vec()),
        dtype,
        device,
        requires_grad,
    )
}

/// Determinant of every matrix in a stack of square matrices.
///
/// The result has the input's batch shape: a `[n, n]` input gives a scalar, a
/// `[b, n, n]` input a `[b]` vector.
///
/// Prefer [`slogdet`] for anything larger than a few dozen rows -- see the
/// module note on overflow.
pub fn det(tensor: &Tensor) -> Result<Tensor> {
    let (batch_dims, n) = square_layout(tensor, "det")?;
    let batch = batch_dims.iter().product::<usize>().max(1);

    let (f32_out, f64_out) = match factorise(tensor, batch, n)? {
        Determinants::F32 { det, .. } => (det, Vec::new()),
        Determinants::F64 { det, .. } => (Vec::new(), det),
    };
    let mut output = batch_tensor(
        &batch_dims,
        tensor.dtype(),
        tensor.device(),
        f32_out,
        f64_out,
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(DetBackward {
            input: tensor.detach(),
            determinant: output.detach(),
            input_id: tensor.id(),
            ids: [tensor.id()],
        });
        output = with_grad_fn(output, grad_fn)?;
    }
    Ok(output)
}

/// `(sign, log|det|)`, which is the determinant in the form that does not
/// overflow.
///
/// `sign * exp(logabsdet)` is the determinant; a singular matrix reports sign
/// `0` and `-inf`. Only the log part carries a gradient -- the sign is locally
/// constant wherever it is defined.
pub fn slogdet(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch_dims, n) = square_layout(tensor, "slogdet")?;
    let batch = batch_dims.iter().product::<usize>().max(1);
    let dtype = tensor.dtype();
    let device = tensor.device();

    let (sign32, sign64, log32, log64) = match factorise(tensor, batch, n)? {
        Determinants::F32 { sign, log_abs, .. } => (sign, Vec::new(), log_abs, Vec::new()),
        Determinants::F64 { sign, log_abs, .. } => (Vec::new(), sign, Vec::new(), log_abs),
    };

    let sign = batch_tensor(&batch_dims, dtype, device, sign32, sign64, false);
    let mut logabsdet = batch_tensor(
        &batch_dims,
        dtype,
        device,
        log32,
        log64,
        tensor.requires_grad(),
    );

    if logabsdet.requires_grad() {
        let grad_fn = Arc::new(SlogdetBackward {
            input: tensor.detach(),
            input_id: tensor.id(),
            ids: [tensor.id()],
        });
        logabsdet = with_grad_fn(logabsdet, grad_fn)?;
    }
    Ok((sign, logabsdet))
}

/// Matrix inverse, solved rather than factorised again.
///
/// `A @ X = I` through [`solve`], so the pivoting, the batching, the singular
/// check and the gradient are all the ones that were already there.
pub fn inv(tensor: &Tensor) -> Result<Tensor> {
    let (batch_dims, n) = square_layout(tensor, "inv")?;
    let identity = {
        // Built without grad: it is a constant, and `solve` would otherwise
        // record a gradient path back to a tensor nobody holds.
        let _guard = NoGradGuard::new();
        eye_like(tensor, &batch_dims, n)?
    };
    solve(tensor, &identity)
}

/// A stack of identity matrices shaped like `tensor`'s batch.
fn eye_like(tensor: &Tensor, batch_dims: &[usize], n: usize) -> Result<Tensor> {
    let batch = batch_dims.iter().product::<usize>().max(1);
    let mut dims = batch_dims.to_vec();
    dims.push(n);
    dims.push(n);
    let shape = Shape::new(dims);
    let mut data = TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("inv: failed to build the identity")
            })?;
            for b in 0..batch {
                for i in 0..n {
                    slice[b * n * n + i * n + i] = 1.0;
                }
            }
        }
        _ => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("inv: failed to build the identity")
            })?;
            for b in 0..batch {
                for i in 0..n {
                    slice[b * n * n + i * n + i] = 1.0;
                }
            }
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
