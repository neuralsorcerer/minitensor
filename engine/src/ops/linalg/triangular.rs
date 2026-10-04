// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::par_out_chunks;
use crate::tensor::Shape;
use std::mem::MaybeUninit;

/// Rows and columns in one tile of a transpose.
///
/// A transpose reads down columns or writes down them, so one side of it
/// touches a new cache line per element. Square tiles bound that to `TILE`
/// lines a side, each reused `TILE` times: a 256x256 float32 matrix took 166us
/// written column by column and 27 in tiles, and on one thread 1024x1024 took
/// 3.97ms and 0.62 -- where the column gather it replaced there took 0.57ms on
/// four. 16 was the best of 8 to 64 across float32 and float64 and shapes from
/// square to 64x4096, and tiles won from 32x32 up.
const TRANSPOSE_TILE: usize = 16;

/// Transpose `input_data` (viewed as `input_shape`) into a fresh contiguous
/// buffer of `output_shape` with `dim0`/`dim1` swapped. Every output element
/// is written exactly once (no zeroing pass; see `ops::map`).
///
/// The input is read as `[outer, lo, between, hi, inner]` -- the two swapped
/// axes, what lies between them and what follows -- and the output is
/// `[outer, hi, between, lo, inner]`. Two shapes of work cover every case:
///
/// - `inner > 1`: the trailing axes move together, so each output run of
///   `inner` elements is one contiguous run of the input and is copied whole.
///   Walking them element by element took 80us to swap the leading axes of an
///   (8, 64, 64) float32 tensor, where this takes 4.4.
/// - `inner == 1`: the swapped axes are the two sides of a matrix, one for
///   each position of `outer` and `between`, and are transposed in tiles; see
///   [`TRANSPOSE_TILE`]. `between` is 1 for `.T` of a matrix and `.mT` of a
///   batch; above it, a (32, 32, 32) `transpose(0, 2)` took 72us element by
///   element and takes 20 in tiles.
pub(crate) fn transpose_map<T: Copy + Send + Sync>(
    input_data: &[T],
    input_shape: &Shape,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Vec<T> {
    let numel = output_shape.numel();
    if numel == 0 {
        return Vec::new();
    }
    let dims = input_shape.dims();
    let (lo, hi) = (dim0.min(dim1), dim0.max(dim1));
    let (d_lo, d_hi) = (dims[lo], dims[hi]);
    let between: usize = dims[lo + 1..hi].iter().product();
    let inner: usize = dims[hi + 1..].iter().product();
    assert_eq!(input_data.len(), numel);

    if inner > 1 {
        // Output run `((a * d_hi + j) * between + b) * d_lo + i` is input run
        // `((a * d_lo + i) * between + b) * d_hi + j`.
        let work = |start: usize, out: &mut [MaybeUninit<T>]| {
            let run = start / inner;
            let (mut i, rest) = (run % d_lo, run / d_lo);
            let (mut b, rest) = (rest % between, rest / between);
            let (mut j, mut a) = (rest % d_hi, rest / d_hi);
            for slot in out.chunks_exact_mut(inner) {
                let from = (((a * d_lo + i) * between + b) * d_hi + j) * inner;
                slot.write_copy_of_slice(&input_data[from..from + inner]);
                i += 1;
                if i == d_lo {
                    i = 0;
                    b += 1;
                    if b == between {
                        b = 0;
                        j += 1;
                        if j == d_hi {
                            j = 0;
                            a += 1;
                        }
                    }
                }
            }
        };
        return fill_transpose(numel, inner, &work);
    }

    // One output "pair" is a position of `outer` and of `hi`: `between` rows
    // of `d_lo`, the rows a tile of `hi` positions writes into.
    let pair_len = between * d_lo;
    let src_stride = between * d_hi;
    let work = |start: usize, out: &mut [MaybeUninit<T>]| {
        let (first, count) = (start / pair_len, out.len() / pair_len);
        assert_eq!(out.len(), count * pair_len);
        let (src, dst) = (input_data.as_ptr(), out.as_mut_ptr());
        // The task's pairs, one position of `outer` at a time, swept a band
        // of `lo` rows at a time: each band of input rows is read across
        // before the next, which ran 10% faster than tile-columns first.
        let mut pair = first;
        while pair < first + count {
            let (a, j_start) = (pair / d_hi, pair % d_hi);
            let j_end = d_hi.min(j_start + first + count - pair);
            for b in 0..between {
                let src_base = a * d_lo * src_stride + b * d_hi;
                for i0 in (0..d_lo).step_by(TRANSPOSE_TILE) {
                    let i1 = (i0 + TRANSPOSE_TILE).min(d_lo);
                    for j0 in (j_start..j_end).step_by(TRANSPOSE_TILE) {
                        for j in j0..(j0 + TRANSPOSE_TILE).min(j_end) {
                            let out_row = ((pair - first + j - j_start) * between + b) * d_lo;
                            for i in i0..i1 {
                                // SAFETY: `pair - first + j - j_start < count`
                                // and `b < between`, `i < d_lo`, so the write
                                // is inside `out`, whose length is asserted
                                // above; the read is input position
                                // (a, i, b, j), each below its extent, and the
                                // input's length is asserted to be `numel`.
                                // Indexing checked both and ran 1.7x slower.
                                unsafe {
                                    (*dst.add(out_row + i))
                                        .write(*src.add(src_base + i * src_stride + j))
                                };
                            }
                        }
                    }
                }
            }
            pair += j_end - j_start;
        }
    };
    fill_transpose(numel, pair_len * TRANSPOSE_TILE, &work)
}

/// A fresh buffer of `numel` filled by `work` -- in tasks of whole multiples
/// of `unit`, widened to 8K elements, above the parallel threshold.
///
/// Small tasks and many of them, because the caller works through them while
/// the pool wakes: a 4096x64 matrix is only four bands of tiles.
fn fill_transpose<T: Send>(
    numel: usize,
    unit: usize,
    work: &(dyn Fn(usize, &mut [MaybeUninit<T>]) + Sync),
) -> Vec<T> {
    let task = unit * (1usize << 13).div_ceil(unit);
    // SAFETY: the tasks tile the output, and `work` writes every element of
    // the whole units it is given.
    unsafe {
        crate::ops::map::build_vec_with::<T, std::convert::Infallible, _>(numel, |spare| {
            if numel < PAR_THRESHOLD {
                work(0, spare);
            } else {
                par_out_chunks(spare, task, work);
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;
    use crate::tensor::Tensor;
    use crate::test_support::{tensor_of, tensor_on};
    use crate::{autograd::GradientFunction, device::Device, tensor::TensorData};
    use std::sync::Arc;

    fn create_test_tensor_f64(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float64);

        if let Some(slice) = tensor_data.as_f64_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        )
    }

    fn create_test_tensor_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);

        if let Some(slice) = tensor_data.as_i32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn create_test_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Bool);

        if let Some(slice) = tensor_data.as_bool_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_matmul_basic() {
        // 2x3 * 3x2 = 2x2
        let a = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let b = tensor_of::<f32>(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], false);

        let result = matmul(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64; 139, 154]
        assert_eq!(result_data, &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_matmul_i32_zero_k_dimension() {
        let a = create_test_tensor_i32(vec![], vec![2, 0]);
        let b = create_test_tensor_i32(vec![], vec![0, 3]);

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
        assert_eq!(result.data().as_i32_slice().unwrap(), &[0, 0, 0, 0, 0, 0]);
    }

    /// Textbook `i`/`j`/`l` reference for the reordered integer kernel.
    fn reference_matmul_i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    #[test]
    fn test_matmul_i32_matches_reference_across_par_threshold() {
        // `m * n * k` straddles PAR_THRESHOLD (4096) so both the sequential and
        // the row-parallel branch of the reordered kernel are exercised.
        for &(m, k, n) in &[(2usize, 3usize, 4usize), (7, 5, 9), (16, 17, 18)] {
            let a: Vec<i32> = (0..m * k).map(|x| (x as i32 % 11) - 5).collect();
            let b: Vec<i32> = (0..k * n).map(|x| (x as i32 % 7) - 3).collect();
            let ta = create_test_tensor_i32(a.clone(), vec![m, k]);
            let tb = create_test_tensor_i32(b.clone(), vec![k, n]);

            let got = matmul(&ta, &tb).unwrap();
            assert_eq!(got.shape().dims(), &[m, n]);
            assert_eq!(
                got.data().as_i32_slice().unwrap(),
                reference_matmul_i32(&a, &b, m, k, n).as_slice(),
                "matmul {m}x{k} @ {k}x{n}"
            );
        }
    }

    #[test]
    fn test_matmul_i32_batched_matches_reference() {
        let (batch, m, k, n) = (3usize, 4usize, 5usize, 6usize);
        let a: Vec<i32> = (0..batch * m * k).map(|x| (x as i32 % 9) - 4).collect();
        let b: Vec<i32> = (0..batch * k * n).map(|x| (x as i32 % 5) - 2).collect();
        let ta = create_test_tensor_i32(a.clone(), vec![batch, m, k]);
        let tb = create_test_tensor_i32(b.clone(), vec![batch, k, n]);

        let got = matmul(&ta, &tb).unwrap();
        assert_eq!(got.shape().dims(), &[batch, m, n]);

        let mut expected = Vec::with_capacity(batch * m * n);
        for bi in 0..batch {
            expected.extend(reference_matmul_i32(
                &a[bi * m * k..(bi + 1) * m * k],
                &b[bi * k * n..(bi + 1) * k * n],
                m,
                k,
                n,
            ));
        }
        assert_eq!(got.data().as_i32_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_solve_batched_parallel_path_matches_per_batch_solution() {
        // Enough batches to cross into the parallel grouping in `solve_batched`,
        // each an independent 2x2 system with a known solution.
        const BATCH: usize = 64;
        let mut lhs = Vec::with_capacity(BATCH * 4);
        let mut rhs = Vec::with_capacity(BATCH * 2);
        let mut expected = Vec::with_capacity(BATCH * 2);
        for b in 0..BATCH {
            // [[2, 0], [0, s]] x = [2*b, s*(b+1)]  =>  x = [b, b+1]
            let s = (b % 5) as f64 + 1.0;
            lhs.extend_from_slice(&[2.0, 0.0, 0.0, s]);
            rhs.extend_from_slice(&[2.0 * b as f64, s * (b as f64 + 1.0)]);
            expected.extend_from_slice(&[b as f64, b as f64 + 1.0]);
        }

        let a = create_test_tensor_f64(lhs, vec![BATCH, 2, 2], false);
        let b = create_test_tensor_f64(rhs, vec![BATCH, 2], false);
        let x = solve(&a, &b).unwrap();
        assert_eq!(x.shape().dims(), &[BATCH, 2]);
        for (got, want) in x.data().as_f64_slice().unwrap().iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_solve_batched_reports_singular_matrix_in_any_batch() {
        // A singular system anywhere in the batch must fail the whole call,
        // including when it is scheduled on a non-first parallel task.
        const BATCH: usize = 64;
        let mut lhs = Vec::with_capacity(BATCH * 4);
        for b in 0..BATCH {
            if b == BATCH - 1 {
                lhs.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            } else {
                lhs.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            }
        }
        let a = create_test_tensor_f64(lhs, vec![BATCH, 2, 2], false);
        let b = create_test_tensor_f64(vec![1.0; BATCH * 2], vec![BATCH, 2], false);
        assert!(solve(&a, &b).is_err());
    }

    #[test]
    fn test_transpose_2d() {
        let a = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = transpose(&a, 0, 1).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result_data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    /// Every transpose against the definition, over the shapes that exercise
    /// the edges of both kernels: tiles cut short on either side, a single
    /// row or column, axes between the swapped two, trailing runs of one and
    /// of several, axes of extent one, and enough elements to split across
    /// the pool with a task boundary falling mid-matrix and mid-run.
    #[test]
    fn a_transpose_puts_every_element_where_the_definition_does() {
        for dims in [
            vec![1, 1],
            vec![1, 37],
            vec![37, 1],
            vec![17, 33],
            vec![64, 48],
            vec![3, 5, 7],
            vec![4, 16, 1],
            vec![4, 16, 16],
            vec![5, 3, 17, 2],
            vec![2, 19, 3, 21, 4],
            vec![9, 130, 131],
            vec![700, 300],
            vec![70, 60, 40],
        ] {
            let rank = dims.len();
            let numel: usize = dims.iter().product();
            let values: Vec<i64> = (0..numel as i64).collect();
            let input = tensor_of::<i64>(values.clone(), dims.clone(), false);
            let strides: Vec<usize> = (0..rank).map(|d| dims[d + 1..].iter().product()).collect();
            for d0 in 0..rank {
                for d1 in d0 + 1..rank {
                    let got = transpose(&input, d0 as isize, d1 as isize).unwrap();
                    let mut out_dims = dims.clone();
                    out_dims.swap(d0, d1);
                    assert_eq!(got.shape().dims(), &out_dims[..]);
                    let got = got.data().as_i64_slice().unwrap();
                    let mut position = vec![0usize; rank];
                    for (flat, value) in got.iter().enumerate() {
                        let mut rest = flat;
                        for d in (0..rank).rev() {
                            position[d] = rest % out_dims[d];
                            rest /= out_dims[d];
                        }
                        position.swap(d0, d1);
                        let from: usize = position.iter().zip(&strides).map(|(p, s)| p * s).sum();
                        assert_eq!(*value, values[from], "{dims:?} ({d0}, {d1}) at {flat}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = tensor_of::<f32>(vec![1.0, 2.0], vec![1, 2], false);
        let b = tensor_of::<f32>(vec![3.0, 4.0, 5.0], vec![3, 1], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_same_dim() {
        let a = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = transpose(&a, 0, 0).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        // Should be unchanged
        assert_eq!(result_data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_gradient_tracking() {
        let a = tensor_of::<f32>(vec![1.0, 2.0], vec![1, 2], true);
        let b = tensor_of::<f32>(vec![3.0, 4.0], vec![2, 1], true);

        let result = matmul(&a, &b).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
    }

    #[test]
    fn test_matmul_dtype_mismatch() {
        let a = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], false);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_device_mismatch() {
        let a = tensor_on::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false, Device::cpu());
        let b = tensor_on::<f32>(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            false,
            Device::cuda(None),
        );

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_bool_error() {
        let a = create_test_tensor_bool(vec![true, false, true, false], vec![2, 2]);
        let b = create_test_tensor_bool(vec![true, true, false, false], vec![2, 2]);

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_vector_operands() {
        // 1-D @ 1-D is a dot product returning a scalar.
        let a = tensor_of::<f32>(vec![1.0, 2.0], vec![2], false);
        let b = tensor_of::<f32>(vec![3.0, 4.0], vec![2], false);
        let dot = matmul(&a, &b).unwrap();
        assert_eq!(dot.shape().dims(), &[] as &[usize]);
        assert_eq!(dot.data().as_f32_slice().unwrap(), &[11.0]);

        // matrix @ vector -> vector.
        let m = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let mv = matmul(&m, &a).unwrap();
        assert_eq!(mv.shape().dims(), &[2]);
        assert_eq!(mv.data().as_f32_slice().unwrap(), &[5.0, 11.0]);

        // 0-D scalars remain invalid operands.
        let s = tensor_of::<f32>(vec![1.0], vec![], false);
        assert!(matmul(&s, &s).is_err());
    }

    #[test]
    fn test_bmm_basic() {
        let a = tensor_of::<f32>(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
            ],
            vec![2, 2, 3],
            false,
        );
        let b = tensor_of::<f32>(
            vec![
                0.5, 1.0, 1.5, 2.0, 2.5, 3.0, // batch 0
                3.5, 4.0, 4.5, 5.0, 5.5, 6.0, // batch 1
            ],
            vec![2, 3, 2],
            false,
        );

        let result = bmm(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        assert_eq!(
            result_data,
            &[11.0, 14.0, 24.5, 32.0, 110.0, 122.0, 150.5, 167.0]
        );
    }

    #[test]
    fn test_bmm_batch_mismatch() {
        let a = tensor_of::<f32>(vec![1.0; 12], vec![2, 2, 3], false);
        let b = tensor_of::<f32>(vec![2.0; 18], vec![3, 3, 2], false);

        let result = bmm(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_bmm_rank_error() {
        let a = tensor_of::<f32>(vec![1.0; 6], vec![2, 3], false);
        let b = tensor_of::<f32>(vec![2.0; 6], vec![1, 3, 2], false);

        let result = bmm(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_diagonal_main() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = diagonal(&tensor, 0, 0, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2]);
    }

    #[test]
    fn test_diagonal_with_offset() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let upper = diagonal(&tensor, 1, 0, 1).unwrap();
        assert_eq!(upper.data().as_f32_slice().unwrap(), &[2.0, 6.0]);

        let lower = diagonal(&tensor, -1, 0, 1).unwrap();
        assert_eq!(lower.data().as_f32_slice().unwrap(), &[4.0]);
    }

    #[test]
    fn test_diagonal_high_dim_shape() {
        let tensor = tensor_of::<f32>((0..24).map(|v| v as f32).collect(), vec![2, 3, 4], false);
        let result = diagonal(&tensor, 0, 1, 2).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_diagonal_backward_gradients() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let grad_output = tensor_of::<f32>(vec![1.0, 1.0], vec![2], false);

        let backward_fn = crate::autograd::DiagonalBackward {
            input_shape: tensor.shape().dims().to_vec(),
            input_strides: tensor.strides().as_slice().to_vec(),
            input_dtype: DataType::Float32,
            dim1: 0,
            dim2: 1,
            offset: 0,
            input_requires_grad: true,
            input_id: tensor.id(),
        };

        let gradients = backward_fn.backward(&grad_output).unwrap();
        let grad_tensor = gradients.get(&tensor.id()).unwrap();
        let grad = grad_tensor.data().as_f32_slice().unwrap();
        assert_eq!(grad, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_trace_matches_manual_sum() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let traced = trace(&tensor, 0, 0, 1).unwrap();
        let value = traced.data().as_f32_slice().unwrap();
        assert_eq!(value, &[5.0]);
    }

    #[test]
    fn test_triu_basic() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_triu_with_positive_diagonal() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = triu(&tensor, 1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tril_basic() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, 0).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tril_with_negative_diagonal() {
        let tensor = tensor_of::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let result = tril(&tensor, -1).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 0.0, 3.0, 0.0]);
    }
}
