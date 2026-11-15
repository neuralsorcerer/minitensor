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
//! [`det`] and [`slogdet`] want more from the elimination than `solve` does --
//! the pivot sign and the diagonal of `U` are what the answer is made of, and
//! `solve` discards both. For a while that was reason enough for a second
//! copy of the elimination to live here. It is not: the factorisation in
//! [`crate::ops::linalg::lu`] keeps what it computed, and both callers read
//! what they need off it. `slogdet` exists next to `det` for the reason it does
//! everywhere: the determinant of a large matrix overflows long before it
//! stops being useful -- a 200x200 matrix of standard normals has a
//! determinant around 1e186, and one twice that size has none that a float64
//! can hold -- while its logarithm is an ordinary number.

use crate::ops::arithmetic::exact_sum::{
    Output, small_cofactors, small_determinant, small_log_determinant, small_solve,
};
use crate::ops::linalg::{EXTRA, Record, Scratch, factor_loaded, solve};
use crate::{
    autograd::{DetBackward, NoGradGuard, SlogdetBackward, with_grad_fn},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// The batch shape and matrix order of a stack of square matrices.
///
/// Shared with [`crate::ops::linalg::cholesky`], which asks the same three
/// questions of its input and must refuse the same inputs in the same words.
pub(crate) fn square_layout(tensor: &Tensor, op: &str) -> Result<(Vec<usize>, usize)> {
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

            let mut record = vec![0i64; n + EXTRA];
            let mut work = Scratch::new();
            for b in 0..batch {
                scratch.copy_from_slice(&data[b * stride..(b + 1) * stride]);
                // The same factorisation `solve` runs, and now literally the
                // same code: the determinant is the product of `U`'s diagonal,
                // negated once per row exchange.
                factor_loaded(&mut scratch, n, &mut record, &mut work)?;
                let negated = Record::new(&record, n).negated();
                let diag: Vec<$ty> = (0..n).map(|i| scratch[i * n + i]).collect();

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
                    log_abs += d.abs().ln();
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
    if (n == 2 || n == 3)
        && let Some(output) = small_det(tensor, &batch_dims, n)?
    {
        // Rounded once from the exact determinant, in the input's dtype.
        return Ok(output);
    }
    // Carried in float64; see `Narrow`.
    if tensor.dtype() == DataType::Float32 {
        return crate::ops::linalg::Narrow::narrowed(det(&crate::ops::linalg::widened(tensor)?)?);
    }
    let batch = batch_dims.iter().product::<usize>();

    let (f32_out, f64_out) = match factorise(tensor, batch, n)? {
        Determinants::F32 { det, .. } => (det, Vec::new()),
        Determinants::F64 { det, .. } => (Vec::new(), det),
    };
    let output = batch_tensor(
        &batch_dims,
        tensor.dtype(),
        tensor.device(),
        f32_out,
        f64_out,
        tensor.requires_grad(),
    );
    with_det_gradient(tensor, output)
}

/// `output`, the determinants of `tensor`, with their gradient attached where
/// one is wanted.
fn with_det_gradient(tensor: &Tensor, output: Tensor) -> Result<Tensor> {
    if !output.requires_grad() {
        return Ok(output);
    }
    let grad_fn = Arc::new(DetBackward {
        input: tensor.detach(),
        determinant: output.detach(),
        input_id: tensor.id(),
        ids: [tensor.id()],
    });
    with_grad_fn(output, grad_fn)
}

/// Every entry of `tensor`, float32 or float64, in float64, which holds a
/// float32 exactly; rows in order.
fn wide_entries(tensor: &Tensor) -> Result<Vec<f64>> {
    let tensor = tensor.contiguous()?;
    let missing = || MinitensorError::internal_error("small matrices: an unexpected dtype");
    Ok(match tensor.dtype() {
        DataType::Float32 => {
            let values = tensor.data().as_f32_slice().ok_or_else(missing)?;
            values.iter().map(|&value| f64::from(value)).collect()
        }
        _ => tensor.data().as_f64_slice().ok_or_else(missing)?.to_vec(),
    })
}

/// Whether every value is finite.
fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// [`det`] of matrices of order two or three: each the float nearest the
/// exact determinant. None where a value is not finite, which elimination
/// answers.
fn small_det(tensor: &Tensor, batch_dims: &[usize], n: usize) -> Result<Option<Tensor>> {
    let entries = wide_entries(tensor)?;
    if !finite(&entries) {
        return Ok(None);
    }
    fn determinants<T: Output>(entries: &[f64], n: usize) -> Vec<T> {
        let mut out = vec![T::narrow(0.0); entries.len() / (n * n)];
        crate::ops::map::par_out_chunks(&mut out, 1 << 10, &|start, part| {
            for (offset, slot) in part.iter_mut().enumerate() {
                *slot = small_determinant(&entries[(start + offset) * n * n..][..n * n], n);
            }
        });
        out
    }
    let (f32_out, f64_out) = match tensor.dtype() {
        DataType::Float32 => (determinants(&entries, n), Vec::new()),
        _ => (Vec::new(), determinants(&entries, n)),
    };
    let output = batch_tensor(
        batch_dims,
        tensor.dtype(),
        tensor.device(),
        f32_out,
        f64_out,
        tensor.requires_grad(),
    );
    with_det_gradient(tensor, output).map(Some)
}

/// The gradient of [`det`] for matrices of order two or three: the incoming
/// gradient times each cofactor, each entry the float nearest its exact
/// value. None for other orders, or where a value is not finite.
pub(crate) fn small_det_gradient(input: &Tensor, grad_output: &Tensor) -> Result<Option<Tensor>> {
    let dims = input.shape().dims();
    let n = dims[dims.len() - 1];
    if !(n == 2 || n == 3) {
        return Ok(None);
    }
    let entries = wide_entries(input)?;
    let scales = wide_entries(grad_output)?;
    if scales.len() * n * n != entries.len() || !finite(&entries) || !finite(&scales) {
        return Ok(None);
    }
    fn gradients<T: Output>(entries: &[f64], scales: &[f64], n: usize) -> Vec<T> {
        let mut out = vec![T::narrow(0.0); entries.len()];
        crate::ops::map::par_out_chunks(&mut out, n * n * 256, &|start, part| {
            for (offset, matrix_out) in part.chunks_mut(n * n).enumerate() {
                let index = start / (n * n) + offset;
                let matrix = &entries[index * n * n..][..n * n];
                small_cofactors(matrix, n, scales[index], matrix_out);
            }
        });
        out
    }
    let data = match input.dtype() {
        DataType::Float32 => {
            TensorData::from_vec_f32(gradients(&entries, &scales, n), input.device())
        }
        _ => TensorData::from_vec_f64(gradients(&entries, &scales, n), input.device()),
    };
    Ok(Some(Tensor::new(
        Arc::new(data),
        input.shape().clone(),
        input.dtype(),
        input.device(),
        false,
    )))
}

/// Whether [`crate::ops::linalg::solve`] takes `lhs x = rhs` by
/// [`small_solutions`]: square float matrices of order two or three, and every
/// value finite.
pub(crate) fn small_order(lhs: &Tensor, rhs: &Tensor) -> Result<bool> {
    let dims = lhs.shape().dims();
    let float = matches!(lhs.dtype(), DataType::Float32 | DataType::Float64);
    if lhs.ndim() < 2 || rhs.ndim() < 1 || !float || lhs.dtype() != rhs.dtype() {
        return Ok(false);
    }
    let n = dims[dims.len() - 1];
    if dims[dims.len() - 2] != n || !(n == 2 || n == 3) {
        return Ok(false);
    }
    Ok(finite(&wide_entries(lhs)?) && finite(&wide_entries(rhs)?))
}

/// The solutions of systems of order two or three into `out`, the
/// operands broadcast against each other: each entry the float nearest its
/// exact value, a quotient of two determinants by Cramer's rule, in the
/// operands' dtype. A system with an exactly singular matrix is refused,
/// however elimination would have rounded its last pivot.
pub(crate) fn small_solutions(
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut TensorData,
    n: usize,
    columns: usize,
) -> Result<()> {
    let (a, b) = (wide_entries(lhs)?, wide_entries(rhs)?);
    fn solutions<T: Output>(
        a: &[f64],
        b: &[f64],
        n: usize,
        columns: usize,
        out: &mut [T],
    ) -> Result<()> {
        let (stride, span) = (n * n, n * columns);
        let one = |index: usize, out: &mut [T]| -> Result<()> {
            let matrix = &a[index * stride..][..stride];
            if small_solve(matrix, n, &b[index * span..][..span], columns, out) {
                return Ok(());
            }
            Err(MinitensorError::invalid_operation(
                "solve received a singular matrix: its determinant is exactly zero",
            ))
        };
        if span == 0 {
            // No columns to write, and every matrix still to be refused if
            // it is singular.
            return (0..a.len() / stride).try_for_each(|index| one(index, &mut []));
        }
        crate::ops::map::try_par_out_chunks(out, span * 256, &|start, part| {
            for (offset, chunk) in part.chunks_mut(span).enumerate() {
                one(start / span + offset, chunk)?;
            }
            Ok(())
        })
    }
    let missing = || MinitensorError::internal_error("solve: an unexpected output dtype");
    match lhs.dtype() {
        DataType::Float32 => solutions(
            &a,
            &b,
            n,
            columns,
            out.as_f32_slice_mut().ok_or_else(missing)?,
        ),
        _ => solutions(
            &a,
            &b,
            n,
            columns,
            out.as_f64_slice_mut().ok_or_else(missing)?,
        ),
    }
}

/// `(sign, log|det|)`, which is the determinant in the form that does not
/// overflow.
///
/// `sign * exp(logabsdet)` is the determinant; a singular matrix reports sign
/// `0` and `-inf`. Only the log part carries a gradient -- the sign is locally
/// constant wherever it is defined.
pub fn slogdet(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
    // Carried in float64; see `Narrow`.
    if tensor.dtype() == DataType::Float32 {
        return crate::ops::linalg::Narrow::narrowed(slogdet(&crate::ops::linalg::widened(
            tensor,
        )?)?);
    }
    let (batch_dims, n) = square_layout(tensor, "slogdet")?;
    let batch = batch_dims.iter().product::<usize>();
    let dtype = tensor.dtype();
    let device = tensor.device();

    // Of order two or three, the sign and the singularity are the exact
    // determinant's, as `det`'s are, and the logarithm is taken of it;
    // elimination stands in for a matrix holding a value that is not
    // finite, and for one whose determinant float64 cannot reach even
    // scaled, and runs only where one does.
    let exact = if n == 2 || n == 3 {
        let entries = wide_entries(tensor)?;
        let mut exact = vec![None; batch];
        crate::ops::map::par_out_chunks(&mut exact, 1 << 10, &|start, part| {
            for (offset, slot) in part.iter_mut().enumerate() {
                let matrix = &entries[(start + offset) * n * n..][..n * n];
                *slot = if finite(matrix) {
                    small_log_determinant(matrix, n)
                } else {
                    None
                };
            }
        });
        exact
    } else {
        Vec::new()
    };
    let (sign32, mut sign64, log32, mut log64) =
        if !exact.is_empty() && exact.iter().all(Option::is_some) {
            let (sign, log) = exact.iter().flatten().copied().unzip();
            (Vec::new(), sign, Vec::new(), log)
        } else {
            match factorise(tensor, batch, n)? {
                Determinants::F32 { sign, log_abs, .. } => (sign, Vec::new(), log_abs, Vec::new()),
                Determinants::F64 { sign, log_abs, .. } => (Vec::new(), sign, Vec::new(), log_abs),
            }
        };
    if sign64.len() == exact.len() {
        for ((sign, log), exact) in sign64.iter_mut().zip(log64.iter_mut()).zip(&exact) {
            if let Some((exact_sign, exact_log)) = *exact {
                *sign = exact_sign;
                *log = exact_log;
            }
        }
    }

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
    // Carried in float64; see `Narrow`.
    if tensor.dtype() == DataType::Float32 {
        return crate::ops::linalg::Narrow::narrowed(inv(&crate::ops::linalg::widened(tensor)?)?);
    }
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
    let batch = batch_dims.iter().product::<usize>();
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
