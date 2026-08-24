// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Eliminating, and substituting.
//!
//! `A = P L U`: the factorisation almost every dense question about a general
//! square matrix reduces to. The determinant is the product of `U`'s diagonal
//! with a sign from the row exchanges; solving `A X = B` is two substitutions
//! once the factorisation is in hand; the inverse is the same against the
//! identity. Doing it once and answering all three from it is the point of this
//! file.
//!
//! It was not done once. `det` carried a partial-pivoting elimination whose own
//! comment said it was "the same ... elimination `solve` runs, kept separate
//! because it answers a different question", and `solve` carried another that
//! threw the factors away as soon as the right-hand side had been updated. Two
//! copies of the same numerics is two places for the substitution order to be
//! wrong and two places to fix it when it is. Both are gone; both callers come
//! through here.
//!
//! ## Partial pivoting
//!
//! At each column the largest remaining entry is brought to the diagonal. That
//! is not a refinement, it is what makes the factorisation usable: without it a
//! matrix as ordinary as `[[0, 1], [1, 0]]` divides by zero at the first step,
//! and one with a merely *small* leading entry produces multipliers large enough
//! to swamp everything computed so far. With it every multiplier is at most one
//! in magnitude.
//!
//! Exchanges are reported the way this library reports every other index: as
//! `int64`, zero-based, and as the row each step swapped with -- not LAPACK's
//! one-based `int32`. `lu_solve` reads them back, so a caller who never looks at
//! them never has to care; one who does gets the convention the rest of the
//! library uses rather than a Fortran inheritance.
//!
//! ## A singular matrix is not always an error
//!
//! A zero pivot means the matrix is singular. For `det` that is the answer --
//! zero -- and the caller asked the question. For `solve` it means there is no
//! solution to return, and a NaN-filled result pretending otherwise would be
//! worse than an error. So the factorisation *records* the first zero pivot and
//! does not decide; the callers decide, differently, and each says so.
//!
//! ## Left-looking
//!
//! One panel of columns at a time, with everything already known subtracted
//! from the panel in a single GEMM before it is factored. The same arithmetic
//! and the same `n^3 / 3` multiply-adds as the right-looking form, but the
//! running correction is rounded once per panel rather than once per panel *per
//! later panel* -- the same argument, and the same shape, as the Cholesky in the
//! next file over.

use crate::{
    error::{MinitensorError, Result},
    ops::{
        linalg::{Factorable, PANEL, ensure, square_layout, transpose},
        map::{PAR_THRESHOLD, try_par_out_chunks, try_par_out_chunks_pair},
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use num_traits::{Float, Zero};
use std::sync::Arc;

/// Which triangle is being solved against, and how it is read.
///
/// Four spellings, one loop. `A X = B` for a lower `A` walks the rows forwards;
/// for an upper `A` it walks them backwards; and transposing swaps which of
/// those applies, because the transpose of a lower triangle is an upper one. So
/// the direction is `upper != transposed`, and the only other difference is
/// whether an element is read at `[i][j]` or at `[j][i]`.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub upper: bool,
    /// Solve against `A^T` without forming it.
    pub transposed: bool,
    /// Treat the diagonal as ones and never read it. The `L` this factorisation
    /// produces is stored that way -- the diagonal it shares is `U`'s.
    pub unit: bool,
}

/// Solve `A X = B` in place, `X` overwriting `B`.
///
/// `a` is `n x n` with row stride `lda`, `x` is `n x cols` with row stride
/// `ldx`. Either may be a block of something larger, which is what lets the
/// panel update below solve directly against the factor it is building.
///
/// A zero on the diagonal is an error rather than an infinity: it means the
/// triangle is singular, and every row that depended on it would otherwise be
/// filled with NaN by the subtraction that follows the divide.
pub(crate) fn substitute<T: Factorable>(
    a: &[T],
    lda: usize,
    n: usize,
    x: &mut [T],
    ldx: usize,
    cols: usize,
    triangle: Triangle,
) -> Result<()> {
    if cols == 0 {
        return Ok(());
    }
    let descending = triangle.upper != triangle.transposed;
    for step in 0..n {
        let i = if descending { n - 1 - step } else { step };
        let (above, from_here) = x.split_at_mut(i * ldx);
        let (current, below) = from_here.split_at_mut(ldx);
        let row = &mut current[..cols];

        // The rows already solved: those below `i` when walking backwards, and
        // those above it when walking forwards.
        let solved = if descending { i + 1..n } else { 0..i };
        for j in solved {
            let coefficient = if triangle.transposed {
                a[j * lda + i]
            } else {
                a[i * lda + j]
            }
            .widen();
            if coefficient == T::Acc::zero() {
                continue;
            }
            let start = if descending {
                (j - i - 1) * ldx
            } else {
                j * ldx
            };
            let source = if descending {
                &below[start..start + cols]
            } else {
                &above[start..start + cols]
            };
            for (slot, value) in row.iter_mut().zip(source) {
                *slot = T::narrow(slot.widen() - coefficient * value.widen());
            }
        }

        if !triangle.unit {
            let pivot = a[i * lda + i].widen();
            if pivot == T::Acc::zero() {
                return Err(MinitensorError::invalid_operation(format!(
                    "triangular solve received a singular matrix: row {i} has a zero on the diagonal"
                )));
            }
            for slot in row.iter_mut() {
                *slot = T::narrow(slot.widen() / pivot);
            }
        }
    }
    Ok(())
}

/// Everything one factorisation produced besides the factors, packed into
/// `n + 2` integers so a batch of them is one contiguous output the parallel
/// split can cut alongside the factors.
///
/// `[0..n]` are the exchanges, `[n]` is whether their number was odd, and
/// `[n + 1]` is one more than the column of the first zero pivot, or zero for a
/// matrix that had none.
pub(crate) const EXTRA: usize = 2;

/// Read one matrix's record.
pub(crate) struct Record<'a> {
    values: &'a [i64],
    n: usize,
}

impl<'a> Record<'a> {
    pub(crate) fn new(values: &'a [i64], n: usize) -> Self {
        Self { values, n }
    }
    pub(crate) fn swaps(&self) -> &'a [i64] {
        &self.values[..self.n]
    }
    pub(crate) fn negated(&self) -> bool {
        self.values[self.n] != 0
    }
    pub(crate) fn singular(&self) -> Option<usize> {
        let marker = self.values[self.n + 1];
        (marker > 0).then(|| (marker - 1) as usize)
    }
}

/// Scratch one task reuses across every matrix it is handed.
///
/// All three stay empty for anything a single panel covers, which is the common
/// case and the one where an allocation per matrix would be most of the cost.
pub(crate) struct Scratch<T> {
    /// The panel's already-factored rows, packed for the substitution and then
    /// reused as the GEMM's right operand -- it is `U12` either way.
    top: Vec<T>,
    /// The finished columns of the rows below the panel, packed so the GEMM can
    /// read them. They sit `n` apart in the factor.
    packed: Vec<T>,
    /// What the GEMM produces, before it is subtracted.
    update: Vec<T>,
}

impl<T: Factorable> Scratch<T> {
    pub(crate) fn new() -> Self {
        Self {
            top: Vec::new(),
            packed: Vec::new(),
            update: Vec::new(),
        }
    }
}

/// Factor the panel spanning columns `[first, first + nb)` over rows
/// `first..n`, exchanging whole rows of `work` as it pivots.
///
/// Whole rows, not just the panel: the columns to the left hold multipliers
/// already computed, and those belong to the row they were computed for. The
/// columns to the right have not been touched yet, so permuting them early
/// costs nothing and saves applying the exchanges again later.
fn factor_panel<T: Factorable>(
    work: &mut [T],
    n: usize,
    first: usize,
    nb: usize,
    record: &mut [i64],
) {
    for k in 0..nb {
        let column = first + k;
        let mut best = column;
        let mut largest = work[column * n + column].widen().abs();
        for row in (column + 1)..n {
            let candidate = work[row * n + column].widen().abs();
            if candidate > largest {
                largest = candidate;
                best = row;
            }
        }

        record[column] = best as i64;
        if best != column {
            for col in 0..n {
                work.swap(column * n + col, best * n + col);
            }
            record[n] ^= 1;
        }

        let pivot = work[column * n + column].widen();
        if pivot == T::Acc::zero() {
            // Singular. There is nothing left to eliminate in this column, and
            // `U` keeps the zero on its diagonal so `det` reports what it must.
            if record[n + 1] == 0 {
                record[n + 1] = column as i64 + 1;
            }
            continue;
        }

        for row in (column + 1)..n {
            let multiplier = work[row * n + column].widen() / pivot;
            work[row * n + column] = T::narrow(multiplier);
            for j in (k + 1)..nb {
                let target = row * n + first + j;
                let above = work[column * n + first + j].widen();
                work[target] = T::narrow(work[target].widen() - multiplier * above);
            }
        }
    }
}

/// Factor the matrix already sitting in `work`, in place, into the packed form:
/// `L` strictly below the diagonal with an implied unit diagonal, `U` on and
/// above it.
pub(crate) fn factor_loaded<T: Factorable>(
    work: &mut [T],
    n: usize,
    record: &mut [i64],
    scratch: &mut Scratch<T>,
) -> Result<()> {
    record.fill(0);
    let mut first = 0usize;
    while first < n {
        let nb = PANEL.min(n - first);
        if first > 0 {
            let rows = n - first;

            // U12 = L11^-1 A12. Packed rather than solved where it sits because
            // L11 and A12 are different columns of the same rows of `work`, and
            // one of the two has to be a copy for the borrow to be legal. A12
            // is the smaller.
            ensure(&mut scratch.top, first * nb);
            for i in 0..first {
                let src = i * n + first;
                scratch.top[i * nb..i * nb + nb].copy_from_slice(&work[src..src + nb]);
            }
            substitute(
                &*work,
                n,
                first,
                &mut scratch.top,
                nb,
                nb,
                Triangle {
                    upper: false,
                    transposed: false,
                    unit: true,
                },
            )?;
            for i in 0..first {
                let dst = i * n + first;
                work[dst..dst + nb].copy_from_slice(&scratch.top[i * nb..i * nb + nb]);
            }

            // A22 -= L21 @ U12, in one GEMM rather than `first` rank-one
            // updates -- which is the whole reason for panelling.
            ensure(&mut scratch.packed, rows * first);
            for i in 0..rows {
                let src = (first + i) * n;
                scratch.packed[i * first..(i + 1) * first].copy_from_slice(&work[src..src + first]);
            }
            ensure(&mut scratch.update, rows * nb);
            // SAFETY: `packed` holds `rows * first` elements, `top` holds
            // `first * nb`, and `update` holds `rows * nb`.
            unsafe {
                T::gemm(
                    rows,
                    first,
                    nb,
                    scratch.packed.as_ptr(),
                    scratch.top.as_ptr(),
                    scratch.update.as_mut_ptr(),
                );
            }
            for i in 0..rows {
                let dst = (first + i) * n + first;
                let target = &mut work[dst..dst + nb];
                let source = &scratch.update[i * nb..i * nb + nb];
                for (slot, correction) in target.iter_mut().zip(source) {
                    *slot = *slot - *correction;
                }
            }
        }

        factor_panel(work, n, first, nb, record);
        first += nb;
    }
    Ok(())
}

/// Apply the exchanges the factorisation made, in the order it made them.
pub(crate) fn apply_swaps<T: Copy>(swaps: &[i64], x: &mut [T], ldx: usize, cols: usize) {
    for (step, &row) in swaps.iter().enumerate() {
        let other = row as usize;
        if other != step {
            for c in 0..cols {
                x.swap(step * ldx + c, other * ldx + c);
            }
        }
    }
}

/// Factor `matrix` and solve `matrix X = rhs` for `X`, both in place.
///
/// The single entry point `solve` uses. `matrix` is destroyed -- it becomes the
/// factorisation -- which is what makes this one pass over the data rather than
/// a factorisation followed by a copy.
pub(crate) fn factor_and_solve<T: Factorable>(
    matrix: &mut [T],
    rhs: &mut [T],
    n: usize,
    cols: usize,
) -> Result<()> {
    let mut record = vec![0i64; n + EXTRA];
    let mut scratch = Scratch::new();
    factor_loaded(matrix, n, &mut record, &mut scratch)?;
    if let Some(column) = Record::new(&record, n).singular() {
        return Err(MinitensorError::invalid_operation(format!(
            "solve received a singular matrix: elimination reached a zero pivot at column {column}"
        )));
    }
    apply_swaps(&record[..n], rhs, cols, cols);
    substitute(
        matrix,
        n,
        n,
        rhs,
        cols,
        cols,
        Triangle {
            upper: false,
            transposed: false,
            unit: true,
        },
    )?;
    substitute(
        matrix,
        n,
        n,
        rhs,
        cols,
        cols,
        Triangle {
            upper: true,
            transposed: false,
            unit: false,
        },
    )
}

/// The packed factorisation of every matrix in a stack, with its records.
pub(crate) struct Packed {
    pub factors: Tensor,
    pub records: Vec<i64>,
    pub n: usize,
    pub batch: usize,
    pub batch_dims: Vec<usize>,
}

impl Packed {
    pub(crate) fn record(&self, index: usize) -> Record<'_> {
        let width = self.n + EXTRA;
        Record::new(&self.records[index * width..(index + 1) * width], self.n)
    }
}

/// Factor a stack of square matrices, whatever their dtype.
pub(crate) fn packed_lu(tensor: &Tensor, op: &str) -> Result<Packed> {
    let (batch_dims, n) = square_layout(tensor, op)?;
    let batch = batch_dims.iter().product::<usize>().max(1);
    let contiguous = tensor.contiguous()?;
    let shape = tensor.shape().clone();
    let mut data = TensorData::zeros_on_device(shape.numel(), tensor.dtype(), tensor.device());
    let width = n + EXTRA;
    let mut records = vec![0i64; batch * width];

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident) => {{
            let src = contiguous.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the input")
            })?;
            let dst = data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the output")
            })?;
            dst.copy_from_slice(src);
            let stride = n * n;
            if stride > 0 {
                try_par_out_chunks_pair(
                    dst,
                    stride,
                    &mut records,
                    width,
                    batch,
                    (PAR_THRESHOLD / (n * n * n / 3).max(1)).clamp(1, batch),
                    &|_first, block_group, record_group| {
                        let mut scratch = Scratch::new();
                        for local in 0..block_group.len() / stride {
                            factor_loaded(
                                &mut block_group[local * stride..(local + 1) * stride],
                                n,
                                &mut record_group[local * width..(local + 1) * width],
                                &mut scratch,
                            )?;
                        }
                        Ok(())
                    },
                )?;
            }
        }};
    }

    match tensor.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut),
        _ => run!(as_f64_slice, as_f64_slice_mut),
    }

    Ok(Packed {
        factors: Tensor::new(
            Arc::new(data),
            shape,
            tensor.dtype(),
            tensor.device(),
            false,
        ),
        records,
        n,
        batch,
        batch_dims,
    })
}

/// The exchanges as an `int64` tensor of shape `(..., n)`.
fn swaps_tensor(packed: &Packed, device: crate::device::Device) -> Result<Tensor> {
    let mut dims = packed.batch_dims.clone();
    dims.push(packed.n);
    let total = packed.batch * packed.n;
    let mut data = TensorData::zeros_on_device(total, DataType::Int64, device);
    if total > 0 {
        let out = data
            .as_i64_slice_mut()
            .ok_or_else(|| MinitensorError::internal_error("lu: pivots are not int64"))?;
        for index in 0..packed.batch {
            let record = packed.record(index);
            for (step, &row) in record.swaps().iter().enumerate() {
                out[index * packed.n + step] = row;
            }
        }
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(dims),
        DataType::Int64,
        device,
        false,
    ))
}

/// The packed factorisation of a stack of square matrices, and its exchanges.
///
/// `L` is unit lower triangular and lives strictly below the diagonal of the
/// first result; `U` is on and above it. The exchanges are `int64` and
/// zero-based: step `i` swapped row `i` with row `pivots[..., i]`.
///
/// The factors come back detached. The gradient of a pivoted factorisation is
/// not implemented here -- the differentiable ways to ask about these matrices
/// are `solve`, `det`, `slogdet` and `inv`, which carry their own gradients and
/// share this very factorisation.
pub fn lu_factor(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
    let packed = packed_lu(tensor, "lu_factor")?;
    let pivots = swaps_tensor(&packed, tensor.device())?;
    Ok((packed.factors, pivots))
}

/// The factorisation spelled out: `P`, `L` and `U` with `A = P @ L @ U`.
///
/// The packed form of [`lu_factor`] is what every consumer here actually wants;
/// this is for a caller who wants to look at the factors, and it is built from
/// that form rather than computed separately.
pub fn lu(tensor: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let packed = packed_lu(tensor, "lu")?;
    let (n, batch) = (packed.n, packed.batch);
    let device = tensor.device();
    let dtype = tensor.dtype();
    let shape = tensor.shape().clone();

    let mut lower = TensorData::zeros_on_device(shape.numel(), dtype, device);
    let mut upper = TensorData::zeros_on_device(shape.numel(), dtype, device);
    let mut permutation = TensorData::zeros_on_device(shape.numel(), dtype, device);

    macro_rules! split {
        ($accessor:ident, $accessor_mut:ident, $one:expr) => {{
            let packedf = packed.factors.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the factor")
            })?;
            let l = lower.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the output")
            })?;
            for index in 0..batch {
                let base = index * n * n;
                for i in 0..n {
                    for j in 0..n {
                        if j < i {
                            l[base + i * n + j] = packedf[base + i * n + j];
                        }
                    }
                    l[base + i * n + i] = $one;
                }
            }
            let u = upper.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the output")
            })?;
            for index in 0..batch {
                let base = index * n * n;
                for i in 0..n {
                    for j in i..n {
                        u[base + i * n + j] = packedf[base + i * n + j];
                    }
                }
            }
            // P undoes the exchanges: row `i` of `A` is row `order[i]` of `L U`,
            // so `P[order[i]][i]` is one. Building it by permuting the identity
            // with the same routine the solve uses is what keeps the two from
            // disagreeing about the direction.
            let p = permutation.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("lu: dtype does not match the output")
            })?;
            for index in 0..batch {
                let base = index * n * n;
                let mut order: Vec<usize> = (0..n).collect();
                let record = packed.record(index);
                for (step, &row) in record.swaps().iter().enumerate() {
                    order.swap(step, row as usize);
                }
                for (i, &source) in order.iter().enumerate() {
                    p[base + source * n + i] = $one;
                }
            }
        }};
    }

    match dtype {
        DataType::Float32 => split!(as_f32_slice, as_f32_slice_mut, 1.0f32),
        _ => split!(as_f64_slice, as_f64_slice_mut, 1.0f64),
    }

    Ok((
        Tensor::new(Arc::new(permutation), shape.clone(), dtype, device, false),
        Tensor::new(Arc::new(lower), shape.clone(), dtype, device, false),
        Tensor::new(Arc::new(upper), shape, dtype, device, false),
    ))
}

/// The right-hand side of a batched solve, checked against the matrices.
///
/// One validator for all three solves below, because they take the same shapes
/// and should refuse the same inputs in the same words. `rhs` is either the
/// same rank as the matrices, `(..., n, k)`, or one less, `(..., n)`, which is
/// the single-column case spelled without the trailing one.
struct Systems {
    cols: usize,
    batch: usize,
    shape: Shape,
}

fn systems(rhs: &Tensor, n: usize, batch_dims: &[usize], op: &str) -> Result<Systems> {
    let rank = rhs.ndim();
    let matrix_rank = batch_dims.len() + 2;
    let dims = rhs.shape().dims();
    let (cols, rhs_batch) = if rank == matrix_rank {
        (dims[rank - 1], &dims[..rank - 2])
    } else if rank + 1 == matrix_rank {
        (1usize, &dims[..rank - 1])
    } else {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects the right-hand side to have the matrices' rank or one less"
        )));
    };
    let rows = if rank == matrix_rank {
        dims[rank - 2]
    } else {
        dims[rank - 1]
    };
    if rows != n {
        return Err(MinitensorError::shape_mismatch(vec![n], vec![rows]));
    }
    if rhs_batch != batch_dims {
        return Err(MinitensorError::shape_mismatch(
            batch_dims.to_vec(),
            rhs_batch.to_vec(),
        ));
    }
    Ok(Systems {
        cols,
        batch: batch_dims.iter().product::<usize>().max(1),
        shape: rhs.shape().clone(),
    })
}

/// Check that two operands can be used together at all.
fn agree(left: &Tensor, right: &Tensor) -> Result<()> {
    if left.device() != right.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", left.device()),
            format!("{:?}", right.device()),
        ));
    }
    if left.dtype() != right.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", left.dtype()),
            format!("{:?}", right.dtype()),
        ));
    }
    Ok(())
}

/// Solve `A X = B` for triangular `A`, without forming `A`'s inverse.
///
/// Only the named triangle of `A` is read; whatever is in the other half is
/// ignored rather than checked, which is what lets a caller pass a packed
/// factorisation straight in. `unitriangular` additionally ignores the
/// diagonal and treats it as ones.
///
/// `left = false` solves `X A = B` instead. That is `A^T X^T = B^T`, so it is
/// this same routine on transposes -- composed rather than written twice, which
/// is also how its gradient comes along.
pub fn solve_triangular(
    a: &Tensor,
    b: &Tensor,
    upper: bool,
    left: bool,
    unitriangular: bool,
) -> Result<Tensor> {
    if !left {
        let flipped = solve_triangular(
            &transpose(a, -2, -1)?,
            &transpose(b, -2, -1)?,
            !upper,
            true,
            unitriangular,
        )?;
        return transpose(&flipped, -2, -1);
    }

    agree(a, b)?;
    let (batch_dims, n) = square_layout(a, "solve_triangular")?;
    let found = systems(b, n, &batch_dims, "solve_triangular")?;
    let triangle = Triangle {
        upper,
        transposed: false,
        unit: unitriangular,
    };
    let matrices = a.contiguous()?;
    let sides = b.contiguous()?;
    let mut out = TensorData::zeros_on_device(found.shape.numel(), a.dtype(), a.device());

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident) => {{
            let left_side = matrices.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("solve_triangular: dtype does not match")
            })?;
            let right_side = sides.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error("solve_triangular: dtype does not match")
            })?;
            let target = out.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error("solve_triangular: dtype does not match")
            })?;
            target.copy_from_slice(right_side);
            substitute_batched(left_side, target, n, &found, triangle)?;
        }};
    }
    match a.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut),
        _ => run!(as_f64_slice, as_f64_slice_mut),
    }

    let solution = Tensor::new(Arc::new(out), found.shape, a.dtype(), a.device(), false);
    if !(crate::autograd::is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
        return Ok(solution);
    }
    // The gradient wants `X` as a matrix even when the caller spelled a single
    // column without its trailing one, because it multiplies by `X^T`.
    let as_matrix = |t: &Tensor| -> Result<Tensor> {
        if t.ndim() + 1 == a.ndim() {
            crate::ops::shape_ops::unsqueeze(t, t.ndim() as isize)
        } else {
            Ok(t.clone())
        }
    };
    let grad_fn = Arc::new(crate::autograd::TriangularSolveBackward {
        matrix: a.detach(),
        solution: as_matrix(&solution)?.detach(),
        upper,
        unitriangular,
        input_ids: [a.id(), b.id()],
        input_requires_grad: [a.requires_grad(), b.requires_grad()],
    });
    crate::autograd::with_grad_fn(solution.requires_grad_(true), grad_fn)
}

/// Substitute every system in the batch, in parallel over the batch.
fn substitute_batched<T: Factorable>(
    matrices: &[T],
    rhs: &mut [T],
    n: usize,
    found: &Systems,
    triangle: Triangle,
) -> Result<()> {
    let stride = n * n;
    let span = n * found.cols;
    if span == 0 || stride == 0 {
        return Ok(());
    }
    let per_task = (PAR_THRESHOLD / (n * n * found.cols.max(1)).max(1)).clamp(1, found.batch);
    let group_span = per_task * span;
    try_par_out_chunks(rhs, group_span, &|start, group| {
        let base = (start / group_span) * per_task;
        for local in 0..group.len() / span {
            let matrix = (base + local) * stride;
            substitute(
                &matrices[matrix..matrix + stride],
                n,
                n,
                &mut group[local * span..(local + 1) * span],
                found.cols,
                found.cols,
                triangle,
            )?;
        }
        Ok(())
    })
}

/// Solve `A X = B` given the factorisation [`lu_factor`] produced for `A`.
///
/// The reason the packed form is worth returning: several right-hand sides
/// against the same matrix cost one factorisation and one pair of
/// substitutions each, rather than a full elimination every time.
///
/// Not differentiable, and the reason is worth stating rather than leaving to
/// be discovered: the factors arrive detached from [`lu_factor`], so there is
/// nothing upstream for a gradient to reach. `solve` is the differentiable way
/// to ask this question, and it runs the same factorisation.
pub fn lu_solve(factors: &Tensor, pivots: &Tensor, b: &Tensor) -> Result<Tensor> {
    agree(factors, b)?;
    let (batch_dims, n) = square_layout(factors, "lu_solve")?;
    let found = systems(b, n, &batch_dims, "lu_solve")?;

    if pivots.dtype() != DataType::Int64 {
        return Err(MinitensorError::invalid_operation(
            "lu_solve: the pivots must be an int64 tensor, as lu_factor returns",
        ));
    }
    let expected: Vec<usize> = batch_dims.iter().copied().chain([n]).collect();
    if pivots.shape().dims() != expected.as_slice() {
        return Err(MinitensorError::shape_mismatch(
            expected,
            pivots.shape().dims().to_vec(),
        ));
    }

    let matrices = factors.contiguous()?;
    let sides = b.contiguous()?;
    let exchanges = pivots.contiguous()?;
    let swaps = exchanges
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("lu_solve: pivots are not int64"))?;
    if let Some(&bad) = swaps
        .iter()
        .find(|&&row| row < 0 || row as usize >= n.max(1))
    {
        return Err(MinitensorError::invalid_operation(format!(
            "lu_solve: pivot {bad} is not a row of an {n}-row matrix"
        )));
    }
    let mut out =
        TensorData::zeros_on_device(found.shape.numel(), factors.dtype(), factors.device());

    macro_rules! run {
        ($accessor:ident, $accessor_mut:ident) => {{
            let packed = matrices
                .data()
                .$accessor()
                .ok_or_else(|| MinitensorError::internal_error("lu_solve: dtype does not match"))?;
            let right_side = sides
                .data()
                .$accessor()
                .ok_or_else(|| MinitensorError::internal_error("lu_solve: dtype does not match"))?;
            let target = out
                .$accessor_mut()
                .ok_or_else(|| MinitensorError::internal_error("lu_solve: dtype does not match"))?;
            target.copy_from_slice(right_side);
            let span = n * found.cols;
            if span > 0 {
                for index in 0..found.batch {
                    apply_swaps(
                        &swaps[index * n..(index + 1) * n],
                        &mut target[index * span..(index + 1) * span],
                        found.cols,
                        found.cols,
                    );
                }
            }
            substitute_batched(
                packed,
                target,
                n,
                &found,
                Triangle {
                    upper: false,
                    transposed: false,
                    unit: true,
                },
            )?;
            substitute_batched(
                packed,
                target,
                n,
                &found,
                Triangle {
                    upper: true,
                    transposed: false,
                    unit: false,
                },
            )?;
        }};
    }
    match factors.dtype() {
        DataType::Float32 => run!(as_f32_slice, as_f32_slice_mut),
        _ => run!(as_f64_slice, as_f64_slice_mut),
    }

    Ok(Tensor::new(
        Arc::new(out),
        found.shape,
        factors.dtype(),
        factors.device(),
        false,
    ))
}

/// Solve `A X = B` given the Cholesky factor of `A` rather than `A` itself.
///
/// Two triangular solves and nothing else: `A = L L^T`, so `L Y = B` and then
/// `L^T X = Y`. Written as exactly that composition, which is why it needs no
/// kernel of its own and no gradient of its own -- both come from
/// [`solve_triangular`].
pub fn cholesky_solve(b: &Tensor, factor: &Tensor, upper: bool) -> Result<Tensor> {
    // The lower triangle is solved against first, whichever way the caller
    // holds the factor: `A = L L^T` inverts as `L^-T L^-1` and `A = U^T U` as
    // `U^-1 U^-T`, and in both of those the lower factor is applied to `B`
    // first. So `upper` decides only which of the two transposes is the lower
    // one -- get that backwards and the answer is the inverse of `L^T L`, which
    // is a different matrix and looks entirely plausible.
    let lower = if upper {
        transpose(factor, -2, -1)?
    } else {
        factor.clone()
    };
    let half = solve_triangular(&lower, b, false, true, false)?;
    let raised = transpose(&lower, -2, -1)?;
    solve_triangular(&raised, &half, true, true, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A X` for a dense `A`, so the substitutions can be checked against the
    /// definition rather than against each other.
    fn apply(a: &[f64], n: usize, x: &[f64], cols: usize, transposed: bool) -> Vec<f64> {
        let mut out = vec![0.0; n * cols];
        for i in 0..n {
            for j in 0..n {
                let coefficient = if transposed {
                    a[j * n + i]
                } else {
                    a[i * n + j]
                };
                for c in 0..cols {
                    out[i * cols + c] += coefficient * x[j * cols + c];
                }
            }
        }
        out
    }

    /// A well-conditioned triangle, so a residual reports the substitution's
    /// arithmetic rather than the matrix's conditioning. The off-diagonals stay
    /// inside one: a unit triangle with larger entries amplifies by roughly
    /// their size per row, and six rows of that is enough to move the last
    /// digits on its own.
    fn triangle(n: usize, upper: bool, unit: bool) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        let mut seed = 1.0f64;
        for i in 0..n {
            for j in 0..n {
                seed = (seed * 7.0 + 3.0) % 11.0 - 5.0;
                let inside = if upper { j >= i } else { j <= i };
                if inside {
                    a[i * n + j] = seed / 5.0;
                }
            }
            a[i * n + i] = if unit { 1.0 } else { n as f64 + 2.0 };
        }
        a
    }

    #[test]
    fn every_spelling_of_the_substitution_solves_what_it_says() {
        for &upper in &[false, true] {
            for &transposed in &[false, true] {
                for &unit in &[false, true] {
                    let n = 6;
                    let cols = 3;
                    let a = triangle(n, upper, unit);
                    let b: Vec<f64> = (0..n * cols).map(|i| (i as f64 * 0.37).sin()).collect();
                    let mut x = b.clone();
                    substitute(
                        &a,
                        n,
                        n,
                        &mut x,
                        cols,
                        cols,
                        Triangle {
                            upper,
                            transposed,
                            unit,
                        },
                    )
                    .unwrap();
                    let back = apply(&a, n, &x, cols, transposed);
                    for (got, want) in back.iter().zip(&b) {
                        assert!(
                            (got - want).abs() < 1e-12,
                            "upper={upper} transposed={transposed} unit={unit}: {got} != {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_unit_diagonal_is_never_read() {
        let n = 4;
        let mut a = triangle(n, false, true);
        let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let triangle_spec = Triangle {
            upper: false,
            transposed: false,
            unit: true,
        };
        let mut want = b.clone();
        substitute(&a, n, n, &mut want, 1, 1, triangle_spec).unwrap();
        for i in 0..n {
            a[i * n + i] = 1e9;
        }
        let mut got = b.clone();
        substitute(&a, n, n, &mut got, 1, 1, triangle_spec).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn a_zero_on_the_diagonal_is_an_error_rather_than_an_infinity() {
        let a = [1.0, 0.0, 2.0, 0.0];
        let mut x = [1.0, 1.0];
        let outcome = substitute(
            &a,
            2,
            2,
            &mut x,
            1,
            1,
            Triangle {
                upper: false,
                transposed: false,
                unit: false,
            },
        );
        assert!(outcome.is_err());
    }

    #[test]
    fn the_factorisation_multiplies_back() {
        // A matrix whose first pivot is zero, so the exchange is exercised.
        let n = 3;
        let a = [0.0, 2.0, 1.0, 4.0, 1.0, 3.0, 2.0, 5.0, 1.0];
        let mut work = a;
        let mut record = vec![0i64; n + EXTRA];
        let mut scratch = Scratch::new();
        factor_loaded(&mut work, n, &mut record, &mut scratch).unwrap();
        let found = Record::new(&record, n);
        assert!(found.singular().is_none());

        // Reconstruct L @ U and undo the exchanges to compare against A.
        let mut product = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut total = 0.0;
                for k in 0..n {
                    let l = match k.cmp(&i) {
                        std::cmp::Ordering::Less => work[i * n + k],
                        std::cmp::Ordering::Equal => 1.0,
                        std::cmp::Ordering::Greater => 0.0,
                    };
                    let u = if k <= j { work[k * n + j] } else { 0.0 };
                    total += l * u;
                }
                product[i * n + j] = total;
            }
        }
        for (step, &row) in found.swaps().iter().enumerate().rev() {
            let other = row as usize;
            if other != step {
                for c in 0..n {
                    product.swap(step * n + c, other * n + c);
                }
            }
        }
        for (got, want) in product.iter().zip(&a) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
    }
}
