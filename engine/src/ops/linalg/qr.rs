// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The QR factorisation, `A = Q R` with `Q` orthonormal and `R` upper
//! triangular.
//!
//! `cholesky` needs a positive-definite matrix and `solve` needs a square one.
//! A rectangular matrix had nothing at all -- no least squares, no way to
//! orthonormalise a set of vectors, no stable basis for a subspace. Those are
//! not compositions of anything the library has: `Q` comes out of a sequence of
//! reflections, each one built from the column the previous ones left behind,
//! and no arrangement of `matmul` and `solve` performs that sequence.
//!
//! Householder reflections rather than Gram-Schmidt. Both produce the same `Q`
//! in exact arithmetic and they are not close in floating point: Gram-Schmidt
//! loses orthogonality in proportion to the condition number of `A`, while a
//! product of reflections is orthogonal to working precision whatever `A` was,
//! because each reflection is orthogonal *individually* and the error cannot
//! accumulate into the property. That is the whole reason to prefer it, and it
//! is what the `Q^T Q = I` test measures.
//!
//! The factorisation is never rejected -- a rank-deficient matrix has a QR,
//! with a zero on `R`'s diagonal. It is the *gradient* that needs `R`
//! invertible, and that is where the failure is reported.

use crate::{
    autograd::{QrBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        linalg::{Factorable, reflector},
        map::{PAR_THRESHOLD, try_par_out_chunks_pair},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::Zero;
use std::sync::Arc;

/// How much of `Q` to return.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QrMode {
    /// `Q` is `[m, k]` and `R` is `[k, n]`, with `k = min(m, n)`. The columns of
    /// `Q` span the column space of `A` and nothing else.
    Reduced,
    /// `Q` is `[m, m]` and `R` is `[m, n]`. The extra columns complete `Q` to a
    /// full orthonormal basis -- and are an arbitrary choice among the bases
    /// that complete it, which is why the gradient refuses this shape when
    /// there are extra columns to choose.
    Complete,
}

impl QrMode {
    /// Parse the name the Python layer passes through.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "reduced" => Ok(QrMode::Reduced),
            "complete" => Ok(QrMode::Complete),
            other => Err(MinitensorError::invalid_argument(format!(
                "unknown qr mode {other:?}; expected \"reduced\" or \"complete\""
            ))),
        }
    }
}

/// The batch shape and the two matrix extents of a stack of matrices.
pub(crate) fn matrix_layout(tensor: &Tensor, op: &str) -> Result<(Vec<usize>, usize, usize)> {
    let ndim = tensor.ndim();
    if ndim < 2 {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects at least 2 dimensions"
        )));
    }
    if !matches!(tensor.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} supports only Float32 and Float64 tensors"
        )));
    }
    let dims = tensor.shape().dims();
    Ok((dims[..ndim - 2].to_vec(), dims[ndim - 2], dims[ndim - 1]))
}

/// Reduce `work` to upper triangular form in place, leaving the reflector that
/// produced each column below that column's diagonal.
///
/// This is LAPACK's convention and it is worth keeping: the reflectors are
/// exactly as many numbers as the space `R`'s zeros leave free, so the whole
/// factorisation lives in one `m x n` buffer plus `k` scalars, and `Q` is built
/// from it afterwards rather than carried alongside.
fn householder<T: Factorable>(
    work: &mut [T],
    m: usize,
    n: usize,
    tau: &mut [T],
    reflectors: &mut [T],
    z: &mut [T::Acc],
    blocks: &mut reflector::Blocks<T>,
) {
    let k = m.min(n);
    let mut start = 0;
    while start < k {
        let nb = reflector::PANEL.min(k - start);
        let stop = start + nb;

        // The panel factors itself: each reflector reaches only the columns
        // still inside the panel, which is all the next reflector needs.
        for j in start..stop {
            tau[j] = reflector::make(work, j * n + j, n, m - j);
            let w = &mut reflectors[(j - start) * m..(j - start) * m + m - j];
            reflector::gather(work, j * n + j, n, m - j, w);
            reflector::apply(work, n, j..m, (j + 1)..stop, w, tau[j], z);
        }

        // Then the columns beyond it, all at once. The panel's reflectors
        // compose into `I - V T V^T`, so what was `nb` sweeps of the trailing
        // block becomes three matrix products over it.
        if stop < n {
            if reflector::worth_blocking(m - start, n - stop, nb) {
                reflector::block(work, m, n, start, nb, &tau[start..stop], blocks);
                reflector::apply_block(work, n, m, start, nb, stop..n, true, blocks);
            } else {
                // `reflectors` was filled while the panel factored itself, so
                // no column of `work` is gathered twice.
                for j in start..stop {
                    let w = &reflectors[(j - start) * m..(j - start) * m + m - j];
                    reflector::apply(work, n, j..m, stop..n, w, tau[j], z);
                }
            }
        }
        start = stop;
    }
}

/// Copy the upper triangle out of the reduced `work` buffer.
///
/// Everything on or above the diagonal is `R`; everything below is the stored
/// reflectors, and the output was allocated zeroed, so only the triangle is
/// written.
fn extract_r<T: Copy>(work: &[T], n: usize, r: &mut [T], r_rows: usize) {
    for i in 0..r_rows {
        if i >= n {
            break;
        }
        r[i * n + i..i * n + n].copy_from_slice(&work[i * n + i..i * n + n]);
    }
}

/// Factor one matrix into `q` and `r`.
fn qr_one<T: Factorable>(
    a: &[T],
    m: usize,
    n: usize,
    q: &mut [T],
    q_cols: usize,
    r: &mut [T],
    r_rows: usize,
    scratch: &mut Scratch<T>,
) {
    let k = m.min(n);
    scratch.work.clear();
    scratch.work.extend_from_slice(a);
    scratch.tau.clear();
    scratch.tau.resize(k, T::zero());
    scratch.reflectors.clear();
    scratch
        .reflectors
        .resize(reflector::PANEL * m.max(1), T::zero());
    scratch.z.clear();
    scratch.z.resize(n.max(q_cols).max(1), T::Acc::zero());

    householder(
        &mut scratch.work,
        m,
        n,
        &mut scratch.tau,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );
    extract_r(&scratch.work, n, r, r_rows);
    reflector::accumulate(
        &scratch.work,
        m,
        n,
        &scratch.tau,
        q,
        q_cols,
        &mut scratch.reflectors,
        &mut scratch.z,
        &mut scratch.blocks,
    );
}

/// Scratch a single task reuses across every matrix it is handed.
struct Scratch<T: Factorable> {
    /// The matrix being reduced: `R` above the diagonal, reflectors below.
    work: Vec<T>,
    tau: Vec<T>,
    /// One panel's worth of gathered reflectors, `m` apart. Gathering them once
    /// per panel rather than once per tile is what keeps the strided read out
    /// of the inner loop.
    reflectors: Vec<T>,
    z: Vec<T::Acc>,
    blocks: reflector::Blocks<T>,
}

impl<T: Factorable> Scratch<T> {
    fn new() -> Self {
        Self {
            work: Vec::new(),
            tau: Vec::new(),
            reflectors: Vec::new(),
            z: Vec::new(),
            blocks: reflector::Blocks::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qr_batched<T: Factorable>(
    input: &[T],
    q: &mut [T],
    r: &mut [T],
    m: usize,
    n: usize,
    q_cols: usize,
    r_rows: usize,
    batch: usize,
) {
    let a_stride = m * n;
    let q_stride = m * q_cols;
    let r_stride = r_rows * n;

    // `Result` only so the pair helper can be shared with routines that fail;
    // a factorisation has nothing to reject.
    // Every extent is non-zero here -- the caller checks -- so the group length
    // divides by the stride and there is no empty-group case to special-case.
    let _: Result<()> = try_par_out_chunks_pair(
        q,
        q_stride,
        r,
        r_stride,
        batch,
        (PAR_THRESHOLD / (m * n * n).max(1)).clamp(1, batch),
        &|first, q_group, r_group| {
            let mut scratch = Scratch::new();
            for local in 0..q_group.len() / q_stride {
                let offset = (first + local) * a_stride;
                qr_one(
                    &input[offset..offset + a_stride],
                    m,
                    n,
                    &mut q_group[local * q_stride..(local + 1) * q_stride],
                    q_cols,
                    &mut r_group[local * r_stride..(local + 1) * r_stride],
                    r_rows,
                    &mut scratch,
                );
            }
            Ok(())
        },
    );
}

/// `(Q, R)` for every matrix in a stack, with `A = Q @ R`.
///
/// `Q` has orthonormal columns and `R` is upper triangular. See [`QrMode`] for
/// the two shapes.
///
/// The factorisation itself never fails; a rank-deficient matrix simply has a
/// zero on `R`'s diagonal. The gradient is the part that needs `R` invertible.
pub fn qr(tensor: &Tensor, mode: QrMode) -> Result<(Tensor, Tensor)> {
    let (batch_dims, m, n) = matrix_layout(tensor, "qr")?;
    // A tensor with no batch dimensions holds one matrix; a batch dimension of
    // zero holds none. `product().max(1)` cannot tell those apart and answers
    // one for both, which is right for the first and a read past the end of an
    // empty input for the second.
    let batch = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product::<usize>()
    };
    let k = m.min(n);
    let (q_cols, r_rows) = match mode {
        QrMode::Reduced => (k, k),
        QrMode::Complete => (m, m),
    };

    let mut q_dims = batch_dims.clone();
    q_dims.extend_from_slice(&[m, q_cols]);
    let mut r_dims = batch_dims.clone();
    r_dims.extend_from_slice(&[r_rows, n]);
    let (q_shape, r_shape) = (Shape::new(q_dims), Shape::new(r_dims));

    let contiguous = tensor.contiguous()?;
    let mut q_data = TensorData::zeros_on_device(q_shape.numel(), tensor.dtype(), tensor.device());
    let mut r_data = TensorData::zeros_on_device(r_shape.numel(), tensor.dtype(), tensor.device());

    if m > 0 && n > 0 && batch > 0 {
        macro_rules! factor {
            ($accessor:ident, $accessor_mut:ident) => {{
                let src = contiguous.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the input slice")
                })?;
                let q_out = q_data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the Q slice")
                })?;
                let r_out = r_data.$accessor_mut().ok_or_else(|| {
                    MinitensorError::internal_error("qr: dtype does not match the R slice")
                })?;
                qr_batched(src, q_out, r_out, m, n, q_cols, r_rows, batch);
            }};
        }
        match tensor.dtype() {
            DataType::Float32 => factor!(as_f32_slice, as_f32_slice_mut),
            _ => factor!(as_f64_slice, as_f64_slice_mut),
        }
    }

    let requires_grad = tensor.requires_grad();
    let mut q = Tensor::new(
        Arc::new(q_data),
        q_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );
    let mut r = Tensor::new(
        Arc::new(r_data),
        r_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );

    if requires_grad {
        // The extra columns of a complete `Q` are an arbitrary completion of the
        // basis, so `Q` is not a function of `A` there and there is nothing to
        // differentiate. Saying so at the factorisation is better than handing
        // back a gradient for a choice the caller never made.
        if mode == QrMode::Complete && m > n {
            return Err(MinitensorError::invalid_operation(
                "qr is not differentiable in complete mode when there are more rows than columns; \
                 use mode=\"reduced\"",
            ));
        }
        let (detached_q, detached_r) = (q.detach(), r.detach());
        let node = |from_q: bool| {
            Arc::new(QrBackward {
                input: tensor.detach(),
                q: detached_q.clone(),
                r: detached_r.clone(),
                from_q,
                input_id: tensor.id(),
                ids: [tensor.id()],
            })
        };
        q = with_grad_fn(q, node(true))?;
        r = with_grad_fn(r, node(false))?;
    }
    Ok((q, r))
}
