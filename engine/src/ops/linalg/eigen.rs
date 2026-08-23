// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The symmetric eigendecomposition, `A = V diag(w) V^T`.
//!
//! `cholesky` factors a positive-definite matrix and `qr` an arbitrary one, but
//! neither says anything about *directions*. The eigenvectors of a symmetric
//! matrix are what principal component analysis returns, what whitening a
//! covariance needs, what a spectral norm is the largest of, and what tells a
//! caller whether their matrix is positive definite and by how much. None of it
//! could be assembled here from anything else -- and unlike every other
//! factorisation in this module, it cannot be computed in a finite number of
//! steps at all. Eigenvalues are roots of the characteristic polynomial, so
//! past degree four there is no formula, and what runs instead is an iteration
//! that converges.
//!
//! Two phases, which is how every implementation does it. Householder
//! reflections reduce the matrix to tridiagonal form in a fixed `n - 2` steps --
//! that part *is* finite, and it is the same reflection `qr` uses, applied from
//! both sides so the result stays symmetric. Then implicitly shifted QL
//! iterations chase the off-diagonal to zero, which is the part that converges
//! rather than terminates, and does so cubically once a shift is close.
//!
//! Only the lower triangle is read, as LAPACK does, and eigenvalues come back
//! in ascending order. Eigenvectors are determined only up to sign -- `v` and
//! `-v` are equally valid -- so no convention is imposed and none should be
//! relied on.

use crate::{
    autograd::{EighBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::linalg::square_layout,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::Float;
use std::sync::Arc;

/// Reduce a symmetric matrix to tridiagonal form, accumulating the transform.
///
/// `work` arrives holding the full symmetric matrix and leaves holding nothing
/// the caller needs; `diagonal` and `offdiagonal` come out holding the
/// tridiagonal, and `vectors` the orthogonal `Q` with `A = Q T Q^T`.
///
/// The reflection is applied from both sides at once, which is what keeps the
/// matrix symmetric and halves the work. Writing `H A H` out for
/// `H = I - tau v v^T` gives a rank-two update:
///
/// ```text
/// p = tau A v,   w = p - (tau / 2)(v . p) v,   A <- A - v w^T - w v^T
/// ```
///
/// so each step touches the trailing block once rather than twice, and the
/// result is symmetric by construction rather than by rounding.
fn tridiagonalize<T: Float>(
    work: &mut [T],
    n: usize,
    diagonal: &mut [T],
    offdiagonal: &mut [T],
    vectors: &mut [T],
    want_vectors: bool,
) {
    let half = T::one() / (T::one() + T::one());
    let mut v = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];

    if want_vectors {
        for value in vectors.iter_mut() {
            *value = T::zero();
        }
        for i in 0..n {
            vectors[i * n + i] = T::one();
        }
    }

    for k in 0..n.saturating_sub(2) {
        let rows = n - k - 1;
        let mut norm_sq = T::zero();
        for i in (k + 1)..n {
            let value = work[i * n + k];
            norm_sq = norm_sq + value * value;
        }
        let alpha = work[(k + 1) * n + k];
        // Everything below the subdiagonal is already zero, so the reflector is
        // the identity and this column is done.
        let below = norm_sq - alpha * alpha;
        if below <= T::zero() {
            offdiagonal[k] = alpha;
            continue;
        }

        let norm = norm_sq.sqrt();
        // Away from `alpha`, so `alpha - beta` is a sum of magnitudes rather
        // than the cancellation that would wreck the reflector.
        let beta = if alpha > T::zero() { -norm } else { norm };
        let tau = (beta - alpha) / beta;
        let scale = T::one() / (alpha - beta);
        v[0] = T::one();
        for i in (k + 2)..n {
            v[i - k - 1] = work[i * n + k] * scale;
        }

        // p = tau * A22 * v, reading the trailing block by rows.
        for (row, slot) in p[..rows].iter_mut().enumerate() {
            let base = (k + 1 + row) * n + k + 1;
            let mut total = T::zero();
            for (col, &component) in v[..rows].iter().enumerate() {
                total = total + work[base + col] * component;
            }
            *slot = total * tau;
        }
        // w = p - (tau / 2)(v . p) v, written back over `p`.
        let mut inner = T::zero();
        for (component, &value) in v[..rows].iter().zip(p[..rows].iter()) {
            inner = inner + *component * value;
        }
        let shift = tau * inner * half;
        for (slot, &component) in p[..rows].iter_mut().zip(v[..rows].iter()) {
            *slot = *slot - shift * component;
        }
        // A22 <- A22 - v w^T - w v^T, symmetric by construction.
        for row in 0..rows {
            let base = (k + 1 + row) * n + k + 1;
            let (vr, wr) = (v[row], p[row]);
            for col in 0..rows {
                work[base + col] = work[base + col] - vr * p[col] - wr * v[col];
            }
        }

        // The column and row this reflector zeroed, written analytically.
        offdiagonal[k] = beta;
        work[(k + 1) * n + k] = beta;
        work[k * n + k + 1] = beta;
        for i in (k + 2)..n {
            work[i * n + k] = T::zero();
            work[k * n + i] = T::zero();
        }

        if want_vectors {
            // Q <- Q H, which touches only the columns the reflector spans.
            for row in 0..n {
                let base = row * n + k + 1;
                let mut total = T::zero();
                for (col, &component) in v[..rows].iter().enumerate() {
                    total = total + vectors[base + col] * component;
                }
                let total = total * tau;
                for (col, &component) in v[..rows].iter().enumerate() {
                    vectors[base + col] = vectors[base + col] - total * component;
                }
            }
        }
    }

    for i in 0..n {
        diagonal[i] = work[i * n + i];
    }
    if n >= 2 {
        offdiagonal[n - 2] = work[(n - 1) * n + n - 2];
    }
    if n >= 1 {
        offdiagonal[n - 1] = T::zero();
    }
}

/// `sqrt(a^2 + b^2)` without the overflow the obvious spelling has.
fn hypotenuse<T: Float>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    if a > b {
        let ratio = b / a;
        a * (T::one() + ratio * ratio).sqrt()
    } else if b > T::zero() {
        let ratio = a / b;
        b * (T::one() + ratio * ratio).sqrt()
    } else {
        T::zero()
    }
}

/// Diagonalise a symmetric tridiagonal matrix by implicitly shifted QL.
///
/// This is the part that iterates. Each sweep applies a chain of plane
/// rotations chosen so that the first one encodes a shift close to an
/// eigenvalue, and the rest chase the resulting bulge off the end of the band --
/// which is why the shift never has to be subtracted from the matrix and no
/// accuracy is lost to cancellation when it is close. Convergence is cubic once
/// it is, and the iteration cap exists only so that a matrix that somehow does
/// not converge reports that rather than spinning.
///
/// `vectors` is rotated alongside when it is wanted, so the columns come out
/// matched to the eigenvalues without a second pass.
fn diagonalize<T: Float>(
    diagonal: &mut [T],
    offdiagonal: &mut [T],
    vectors: &mut [T],
    n: usize,
    want_vectors: bool,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let two = T::one() + T::one();
    let epsilon = T::epsilon();

    for l in 0..n {
        let mut iterations = 0usize;
        loop {
            // The first negligible off-diagonal at or after `l` splits the
            // problem; if it is `l` itself, this eigenvalue has converged.
            let mut m = l;
            while m + 1 < n {
                let scale = diagonal[m].abs() + diagonal[m + 1].abs();
                if offdiagonal[m].abs() <= epsilon * scale {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 50 {
                return Err(MinitensorError::invalid_operation(
                    "eigh did not converge; the matrix may contain NaN or infinity",
                ));
            }

            // Wilkinson's shift, as the eigenvalue of the trailing 2x2 nearer
            // to its corner -- written so the subtraction never cancels.
            let mut g = (diagonal[l + 1] - diagonal[l]) / (two * offdiagonal[l]);
            let mut r = hypotenuse(g, T::one());
            let signed = if g >= T::zero() { r.abs() } else { -r.abs() };
            g = diagonal[m] - diagonal[l] + offdiagonal[l] / (g + signed);

            let (mut s, mut c) = (T::one(), T::one());
            let mut p = T::zero();
            let mut split = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let mut f = s * offdiagonal[i];
                let b = c * offdiagonal[i];
                r = hypotenuse(f, g);
                offdiagonal[i + 1] = r;
                if r == T::zero() {
                    // The band separated underneath the bulge; take the
                    // deflation and start the sweep again.
                    diagonal[i + 1] = diagonal[i + 1] - p;
                    offdiagonal[m] = T::zero();
                    split = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = diagonal[i + 1] - p;
                r = (diagonal[i] - g) * s + two * c * b;
                p = s * r;
                diagonal[i + 1] = g + p;
                g = c * r - b;

                if want_vectors {
                    for row in 0..n {
                        let base = row * n;
                        f = vectors[base + i + 1];
                        vectors[base + i + 1] = s * vectors[base + i] + c * f;
                        vectors[base + i] = c * vectors[base + i] - s * f;
                    }
                }
            }
            if split {
                continue;
            }
            diagonal[l] = diagonal[l] - p;
            offdiagonal[l] = g;
            offdiagonal[m] = T::zero();
        }
    }
    Ok(())
}

/// Sort the eigenvalues ascending, carrying their vectors with them.
///
/// The iteration produces them in whatever order it converged, and every
/// caller -- and NumPy, and LAPACK's `syevd` -- expects ascending. Selection
/// sort rather than anything cleverer: `n` is the matrix order, the comparison
/// count is `n^2 / 2` against the `n^3` that produced them, and swapping a pair
/// of columns is the expensive half either way.
fn sort_ascending<T: Float>(diagonal: &mut [T], vectors: &mut [T], n: usize, want_vectors: bool) {
    for i in 0..n {
        let mut best = i;
        for j in (i + 1)..n {
            if diagonal[j] < diagonal[best] {
                best = j;
            }
        }
        if best != i {
            diagonal.swap(i, best);
            if want_vectors {
                for row in 0..n {
                    vectors.swap(row * n + i, row * n + best);
                }
            }
        }
    }
}

macro_rules! eigh_kernel {
    ($name:ident, $ty:ty, $accessor:ident) => {
        /// Decompose every matrix in the batch.
        fn $name(
            input: &Tensor,
            batch: usize,
            n: usize,
            want_vectors: bool,
        ) -> Result<(Vec<$ty>, Vec<$ty>)> {
            let data = input.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("eigh: dtype does not match the input slice")
            })?;
            let stride = n * n;
            let mut values = vec![0 as $ty; batch * n];
            let mut all_vectors = vec![0 as $ty; if want_vectors { batch * stride } else { 0 }];

            let mut work = vec![0 as $ty; stride];
            let mut offdiagonal = vec![0 as $ty; n.max(1)];
            let mut scratch = vec![0 as $ty; stride];

            for b in 0..batch {
                // Only the lower triangle is read; the upper is mirrored from
                // it, so a matrix that is not symmetric is treated as the
                // symmetric one its lower half describes -- LAPACK's contract.
                for i in 0..n {
                    for j in 0..=i {
                        let value = data[b * stride + i * n + j];
                        work[i * n + j] = value;
                        work[j * n + i] = value;
                    }
                }
                let diagonal = &mut values[b * n..(b + 1) * n];
                tridiagonalize(
                    &mut work,
                    n,
                    diagonal,
                    &mut offdiagonal,
                    &mut scratch,
                    want_vectors,
                );
                diagonalize(diagonal, &mut offdiagonal, &mut scratch, n, want_vectors)?;
                sort_ascending(diagonal, &mut scratch, n, want_vectors);
                if want_vectors {
                    all_vectors[b * stride..(b + 1) * stride].copy_from_slice(&scratch);
                }
            }
            Ok((values, all_vectors))
        }
    };
}

eigh_kernel!(eigh_f32, f32, as_f32_slice);
eigh_kernel!(eigh_f64, f64, as_f64_slice);

/// Shared body for the two public entry points.
fn decompose(tensor: &Tensor, want_vectors: bool, op: &str) -> Result<(Tensor, Tensor)> {
    let (batch_dims, n) = square_layout(tensor, op)?;
    // A tensor with no batch dimensions holds one matrix; a batch dimension of
    // zero holds none, and `product().max(1)` cannot tell those apart.
    let batch = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product::<usize>()
    };

    let mut value_dims = batch_dims.clone();
    value_dims.push(n);
    let mut vector_dims = batch_dims;
    vector_dims.extend_from_slice(&[n, n]);
    let value_shape = Shape::new(value_dims);
    let vector_shape = Shape::new(vector_dims);

    let contiguous = tensor.contiguous()?;
    let (value_data, vector_data) = if batch == 0 || n == 0 {
        (
            TensorData::zeros_on_device(value_shape.numel(), tensor.dtype(), tensor.device()),
            TensorData::zeros_on_device(vector_shape.numel(), tensor.dtype(), tensor.device()),
        )
    } else {
        match tensor.dtype() {
            DataType::Float32 => {
                let (values, vectors) = eigh_f32(&contiguous, batch, n, want_vectors)?;
                (
                    TensorData::from_vec_f32(values, tensor.device()),
                    TensorData::from_vec_f32(vectors, tensor.device()),
                )
            }
            _ => {
                let (values, vectors) = eigh_f64(&contiguous, batch, n, want_vectors)?;
                (
                    TensorData::from_vec_f64(values, tensor.device()),
                    TensorData::from_vec_f64(vectors, tensor.device()),
                )
            }
        }
    };

    let requires_grad = tensor.requires_grad();
    let values = Tensor::new(
        Arc::new(value_data),
        value_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );
    let vectors = Tensor::new(
        Arc::new(vector_data),
        vector_shape,
        tensor.dtype(),
        tensor.device(),
        requires_grad,
    );
    Ok((values, vectors))
}

/// Eigenvalues and eigenvectors of every symmetric matrix in a stack.
///
/// Returns `(w, V)` with `w` ascending and `A @ V == V @ diag(w)`. Only the
/// lower triangle of the input is read.
///
/// Eigenvectors are determined up to sign: `v` and `-v` both satisfy the
/// definition, and nothing here picks between them. Compare `|V|`, or compare
/// what `V` does, rather than `V` itself.
pub fn eigh(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
    let (mut values, mut vectors) = decompose(tensor, true, "eigh")?;

    if tensor.requires_grad() {
        // Two outputs and one gradient at a time, so each gets a node and the
        // engine adds what they produce -- exact, because the gradient is
        // linear in the pair. See `EighBackward`.
        let (detached_values, detached_vectors) = (values.detach(), vectors.detach());
        let node = |from_values: bool| {
            Arc::new(EighBackward {
                values: detached_values.clone(),
                vectors: detached_vectors.clone(),
                from_values,
                input_id: tensor.id(),
                ids: [tensor.id()],
            })
        };
        values = with_grad_fn(values, node(true))?;
        vectors = with_grad_fn(vectors, node(false))?;
    }
    Ok((values, vectors))
}

/// The eigenvalues alone, ascending.
///
/// Skips accumulating the rotations, which is most of the work: the iteration
/// itself only touches three diagonals, while carrying `V` through it is the
/// `n^3` part. Worth having as its own entry point rather than as `eigh(a)[0]`
/// for exactly that reason.
pub fn eigvalsh(tensor: &Tensor) -> Result<Tensor> {
    // The gradient needs the eigenvectors even though the caller did not ask
    // for them, so a differentiable call computes them anyway and this becomes
    // the first half of `eigh`. Nothing is lost: it is the same work `eigh`
    // would have done, and the saving was never available to it.
    if tensor.requires_grad() {
        return Ok(eigh(tensor)?.0);
    }
    Ok(decompose(tensor, false, "eigvalsh")?.0)
}
