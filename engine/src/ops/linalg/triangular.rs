// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::ops::map::par_out_chunks;
use crate::tensor::{Shape, Strides};
use std::mem::MaybeUninit;

/// Rows and columns in one tile of a matrix transpose.
///
/// A transpose reads down columns or writes down them, so one side of it
/// touches a new cache line per element. Square tiles bound that to `TILE`
/// lines a side, each reused `TILE` times: a 256x256 float32 matrix took 163us
/// written column by column and 22 in tiles, 1024x1024 3.97ms and 0.62 on one
/// thread -- where the column gather it replaced there took 0.57ms on four. 16
/// was the best of 8 to 64 across float32 and float64 and shapes from square
/// to 64x4096, and tiles won from 32x32 up.
const TRANSPOSE_TILE: usize = 16;

/// Output rows `first..first + out.len() / rows` of the transpose of one
/// `rows x cols` matrix -- input columns, that is -- tile by tile.
fn transpose_band<T: Copy>(
    src: &[T],
    rows: usize,
    cols: usize,
    first: usize,
    out: &mut [MaybeUninit<T>],
) {
    let count = out.len() / rows;
    assert!(src.len() == rows * cols && out.len() == count * rows && first + count <= cols);
    let (src, out) = (src.as_ptr(), out.as_mut_ptr());
    for i0 in (0..rows).step_by(TRANSPOSE_TILE) {
        let i1 = (i0 + TRANSPOSE_TILE).min(rows);
        for j0 in (0..count).step_by(TRANSPOSE_TILE) {
            for j in j0..(j0 + TRANSPOSE_TILE).min(count) {
                for i in i0..i1 {
                    // SAFETY: `i < rows` and `j < count`, and the assertion
                    // above puts `first + j < cols`, so both offsets are in
                    // bounds. Indexing checked each and ran 1.7x slower.
                    unsafe { (*out.add(j * rows + i)).write(*src.add(i * cols + first + j)) };
                }
            }
        }
    }
}

/// Transpose `input_data` (viewed as `input_shape`) into a fresh contiguous
/// buffer of `output_shape` with `dim0`/`dim1` swapped. Every output element
/// is written exactly once (no zeroing pass; see `ops::map`).
pub(crate) fn transpose_map<T: Copy + Send + Sync>(
    input_data: &[T],
    input_shape: &Shape,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Vec<T> {
    let numel = output_shape.numel();
    let rank = input_shape.ndim();

    // The last two axes swapped is a stack of matrix transposes -- `.T` of a
    // matrix, `.mT` of a batch -- done in tiles; see `TRANSPOSE_TILE`.
    if rank >= 2 && dim0.min(dim1) == rank - 2 && dim0.max(dim1) == rank - 1 {
        if numel == 0 {
            return Vec::new();
        }
        let (rows, cols) = (input_shape.dims()[rank - 2], input_shape.dims()[rank - 1]);
        // The output is `numel / rows` rows of `rows`, one per input column,
        // and any whole number of them is a task: one band of tiles, widened
        // to 8K elements where the rows are short. Small tasks and many of
        // them, because the caller works through them while the pool wakes --
        // a 4096x64 matrix is only four bands.
        let band = TRANSPOSE_TILE * (1usize << 13).div_ceil(TRANSPOSE_TILE * rows);
        let work = |start: usize, out: &mut [MaybeUninit<T>]| {
            let (mut row, mut out) = (start / rows, out);
            while !out.is_empty() {
                let (matrix, column) = (row / cols, row % cols);
                let take = (cols - column).min(out.len() / rows);
                let (here, rest) = out.split_at_mut(take * rows);
                let src = &input_data[matrix * rows * cols..(matrix + 1) * rows * cols];
                transpose_band(src, rows, cols, column, here);
                (row, out) = (row + take, rest);
            }
        };
        // SAFETY: every output row belongs to exactly one task, which writes
        // all of it.
        return unsafe {
            crate::ops::map::build_vec_with::<T, std::convert::Infallible, _>(numel, |spare| {
                if numel < PAR_THRESHOLD {
                    work(0, spare);
                } else {
                    par_out_chunks(spare, band * rows, &work);
                }
                Ok(())
            })
            .unwrap_or_else(|e| match e {})
        };
    }

    // General case: gather through the swapped-stride view of the input.
    // The output's element at output coordinates c reads the input at the
    // same coordinates with dim0/dim1 swapped, which is exactly a strided
    // gather with the input's contiguous strides permuted.
    let input_strides = Strides::from_shape(input_shape);
    let in_strides = input_strides.as_slice();
    let out_dims = output_shape.dims();
    let mut gather_strides: Vec<usize> = (0..out_dims.len())
        .map(|dim| {
            let in_dim = if dim == dim0 {
                dim1
            } else if dim == dim1 {
                dim0
            } else {
                dim
            };
            in_strides[in_dim]
        })
        .collect();
    if out_dims.is_empty() {
        gather_strides.clear();
    }
    crate::ops::map::strided_gather(input_data, out_dims, &gather_strides)
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

    /// The tiled transpose against the definition, over the shapes that
    /// exercise its edges: tiles cut short on either side, a single row or
    /// column, a stack of matrices, and enough of them to split across the
    /// pool with a task boundary falling inside a matrix.
    #[test]
    fn a_tiled_transpose_puts_every_element_where_the_definition_does() {
        for dims in [
            vec![1, 1],
            vec![1, 37],
            vec![37, 1],
            vec![17, 33],
            vec![64, 48],
            vec![3, 5, 7],
            vec![4, 16, 16],
            vec![9, 130, 131],
            vec![700, 300],
        ] {
            let rank = dims.len();
            let (rows, cols) = (dims[rank - 2], dims[rank - 1]);
            let numel: usize = dims.iter().product();
            let values: Vec<i64> = (0..numel as i64).collect();
            let input = tensor_of::<i64>(values.clone(), dims.clone(), false);
            let got = transpose(&input, rank as isize - 1, rank as isize - 2).unwrap();
            let mut want_dims = dims.clone();
            want_dims.swap(rank - 2, rank - 1);
            assert_eq!(got.shape().dims(), &want_dims[..]);
            let got = got.data().as_i64_slice().unwrap();
            for matrix in 0..numel / (rows * cols) {
                let base = matrix * rows * cols;
                for i in 0..rows {
                    for j in 0..cols {
                        assert_eq!(
                            got[base + j * rows + i],
                            values[base + i * cols + j],
                            "{dims:?} at ({matrix}, {i}, {j})"
                        );
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
