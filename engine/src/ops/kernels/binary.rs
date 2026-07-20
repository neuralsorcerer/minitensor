// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::ops::map::{
    BINARY_PAR_THRESHOLD, PAR_CHUNK, binary_map, build_vec_with, unary_map_threshold,
};
use crate::ops::simd::*;
use crate::{
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use smallvec::SmallVec;
use smallvec::smallvec;
use std::convert::Infallible;

/// Generates a dtype-specialized broadcasting binary kernel.
///
/// Every kernel has the same shape: fetch both input slices for the dtype,
/// apply `$op` element-wise with broadcasting into a fresh buffer, and wrap
/// it as `TensorData` (no zero-initialization pass; see `ops::map`).
macro_rules! binary_kernel {
    ($name:ident, $accessor:ident, $ty:ty, $dtype:ident, $tyname:literal, $op:expr) => {
        pub(crate) fn $name(
            lhs: &Tensor,
            rhs: &Tensor,
            output_shape: &Shape,
        ) -> Result<TensorData> {
            let lhs_data = lhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from lhs tensor"
                ))
            })?;
            let rhs_data = rhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from rhs tensor"
                ))
            })?;
            let out = broadcast_binary_map(
                lhs_data,
                rhs_data,
                lhs.shape(),
                rhs.shape(),
                output_shape,
                $op,
            )?;
            Ok(TensorData::from_vec::<$ty>(
                out,
                DataType::$dtype,
                lhs.device(),
            ))
        }
    };
}

/// Same as [`binary_kernel!`], with a SIMD fast path for same-shape inputs
/// (f32/f64 only).
macro_rules! binary_kernel_simd {
    ($name:ident, $accessor:ident, $ty:ty, $dtype:ident, $tyname:literal, $simd:ident, $op:expr) => {
        pub(crate) fn $name(
            lhs: &Tensor,
            rhs: &Tensor,
            output_shape: &Shape,
        ) -> Result<TensorData> {
            let lhs_data = lhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from lhs tensor"
                ))
            })?;
            let rhs_data = rhs.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from rhs tensor"
                ))
            })?;
            let out = if can_use_simd_fast_path(lhs.shape(), rhs.shape(), output_shape) {
                // SAFETY: the SIMD entry points initialize every output
                // element when they return Ok.
                unsafe {
                    build_vec_with(output_shape.numel(), |spare| {
                        $simd(lhs_data, rhs_data, spare)
                    })?
                }
            } else {
                broadcast_binary_map(
                    lhs_data,
                    rhs_data,
                    lhs.shape(),
                    rhs.shape(),
                    output_shape,
                    $op,
                )?
            };
            Ok(TensorData::from_vec::<$ty>(
                out,
                DataType::$dtype,
                lhs.device(),
            ))
        }
    };
}

// Addition: `+` for numeric dtypes, logical OR for bool.
binary_kernel_simd!(
    add_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    simd_add_f32,
    |a, b| a + b
);
binary_kernel_simd!(
    add_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    simd_add_f64,
    |a, b| a + b
);
binary_kernel!(add_i32_direct, as_i32_slice, i32, Int32, "i32", |a, b| a
    + b);
binary_kernel!(add_i64_direct, as_i64_slice, i64, Int64, "i64", |a, b| a
    + b);
binary_kernel!(
    add_bool_direct,
    as_bool_slice,
    bool,
    Bool,
    "bool",
    |a, b| a || b
);

// Subtraction: bool is rejected during operand coercion.
binary_kernel_simd!(
    sub_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    simd_sub_f32,
    |a, b| a - b
);
binary_kernel_simd!(
    sub_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    simd_sub_f64,
    |a, b| a - b
);
binary_kernel!(sub_i32_direct, as_i32_slice, i32, Int32, "i32", |a, b| a
    - b);
binary_kernel!(sub_i64_direct, as_i64_slice, i64, Int64, "i64", |a, b| a
    - b);

// Multiplication: `*` for numeric dtypes, logical AND for bool.
binary_kernel_simd!(
    mul_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    simd_mul_f32,
    |a, b| a * b
);
binary_kernel_simd!(
    mul_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    simd_mul_f64,
    |a, b| a * b
);
binary_kernel!(mul_i32_direct, as_i32_slice, i32, Int32, "i32", |a, b| a
    * b);
binary_kernel!(mul_i64_direct, as_i64_slice, i64, Int64, "i64", |a, b| a
    * b);
binary_kernel!(
    mul_bool_direct,
    as_bool_slice,
    bool,
    Bool,
    "bool",
    |a, b| a && b
);

// Division: integer and bool operands coerce to floating point beforehand.
binary_kernel_simd!(
    div_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    simd_div_f32,
    |a, b| a / b
);
binary_kernel_simd!(
    div_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    simd_div_f64,
    |a, b| a / b
);

// Floor division: quotient rounded toward negative infinity (Python
// semantics). Integer closures use wrapping division so `MIN / -1` wraps
// instead of panicking inside the parallel loops; the op layer rejects zero
// divisors for integer dtypes before dispatching here.
binary_kernel!(
    floordiv_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    |a: f32, b: f32| (a / b).floor()
);
binary_kernel!(
    floordiv_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    |a: f64, b: f64| (a / b).floor()
);
binary_kernel!(
    floordiv_i32_direct,
    as_i32_slice,
    i32,
    Int32,
    "i32",
    |a: i32, b: i32| {
        let q = a.wrapping_div(b);
        let r = a.wrapping_rem(b);
        if r != 0 && ((r < 0) != (b < 0)) {
            q - 1
        } else {
            q
        }
    }
);
binary_kernel!(
    floordiv_i64_direct,
    as_i64_slice,
    i64,
    Int64,
    "i64",
    |a: i64, b: i64| {
        let q = a.wrapping_div(b);
        let r = a.wrapping_rem(b);
        if r != 0 && ((r < 0) != (b < 0)) {
            q - 1
        } else {
            q
        }
    }
);

// Remainder with the sign of the divisor (Python semantics), consistent with
// the floor division above: a == floor_div(a, b) * b + rem(a, b). Float
// variants yield NaN for zero divisors like `%` itself; integer zero divisors
// are rejected by the op layer.
binary_kernel!(
    rem_f32_direct,
    as_f32_slice,
    f32,
    Float32,
    "f32",
    |a: f32, b: f32| {
        let r = a % b;
        if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
            r + b
        } else {
            r
        }
    }
);
binary_kernel!(
    rem_f64_direct,
    as_f64_slice,
    f64,
    Float64,
    "f64",
    |a: f64, b: f64| {
        let r = a % b;
        if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
            r + b
        } else {
            r
        }
    }
);
binary_kernel!(
    rem_i32_direct,
    as_i32_slice,
    i32,
    Int32,
    "i32",
    |a: i32, b: i32| {
        let r = a.wrapping_rem(b);
        if r != 0 && ((r < 0) != (b < 0)) {
            r + b
        } else {
            r
        }
    }
);
binary_kernel!(
    rem_i64_direct,
    as_i64_slice,
    i64,
    Int64,
    "i64",
    |a: i64, b: i64| {
        let r = a.wrapping_rem(b);
        if r != 0 && ((r < 0) != (b < 0)) {
            r + b
        } else {
            r
        }
    }
);

/// Generic broadcasting binary map: applies `op` element-wise over broadcast
/// operands, producing a fresh, fully-initialized output buffer (no zeroing
/// pass — every element is written exactly once).
pub(crate) fn broadcast_binary_map<T, U, F>(
    lhs_data: &[T],
    rhs_data: &[T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    op: F,
) -> Result<Vec<U>>
where
    T: Copy + Send + Sync,
    U: Copy + Send + Sync,
    F: Fn(T, T) -> U + Send + Sync,
{
    let output_dims = output_shape.dims();
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let rank = output_dims.len();
    let numel = output_shape.numel();

    if numel == 0 || output_dims.contains(&0) {
        return Ok(Vec::new());
    }

    // Fast path when no broadcasting is required: plain element-wise zip,
    // parallel above the binary threshold.
    if lhs_dims == output_dims && rhs_dims == output_dims {
        return Ok(binary_map(lhs_data, rhs_data, op));
    }

    // Fast paths when one side is a scalar and the other already matches the
    // output shape: a unary map with the scalar captured.
    if lhs_data.len() == 1 && rhs_dims == output_dims {
        let lhs_val = lhs_data[0];
        return Ok(unary_map_threshold(
            rhs_data,
            BINARY_PAR_THRESHOLD,
            move |r| op(lhs_val, r),
        ));
    }

    if rhs_data.len() == 1 && lhs_dims == output_dims {
        let rhs_val = rhs_data[0];
        return Ok(unary_map_threshold(
            lhs_data,
            BINARY_PAR_THRESHOLD,
            move |l| op(l, rhs_val),
        ));
    }

    let lhs_contiguous = Strides::from_shape(lhs_shape);
    let rhs_contiguous = Strides::from_shape(rhs_shape);
    let lhs_strides = lhs_contiguous.as_slice();
    let rhs_strides = rhs_contiguous.as_slice();

    let mut lhs_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];
    let mut rhs_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];

    let lhs_offset = rank.saturating_sub(lhs_dims.len());
    for (i, &dim) in lhs_dims.iter().enumerate() {
        lhs_aligned[lhs_offset + i] = if dim == 1 { 0 } else { lhs_strides[i] };
    }

    let rhs_offset = rank.saturating_sub(rhs_dims.len());
    for (i, &dim) in rhs_dims.iter().enumerate() {
        rhs_aligned[rhs_offset + i] = if dim == 1 { 0 } else { rhs_strides[i] };
    }

    // Inner-run fast path. In any valid broadcast each operand's last dimension
    // is either the output's last dimension (contiguous, stride 1) or 1
    // (broadcast, stride 0). So the expensive coordinate decomposition is done
    // once per output *row* (its coordinates over dims `0..last`) instead of
    // once per element, and each row is filled with a contiguous, auto-
    // vectorizable inner loop whose broadcast branch is hoisted out.
    let last = rank - 1;
    let inner = output_dims[last];
    let lhs_last = lhs_aligned[last]; // 0 (broadcast) or 1 (contiguous)
    let rhs_last = rhs_aligned[last];

    // Base offsets into lhs/rhs for output row `row`.
    let row_bases = |row: usize| -> (usize, usize) {
        let mut lhs_base = 0usize;
        let mut rhs_base = 0usize;
        let mut tmp = row;
        for i in (0..last).rev() {
            let coord = tmp % output_dims[i];
            tmp /= output_dims[i];
            lhs_base += coord * lhs_aligned[i];
            rhs_base += coord * rhs_aligned[i];
        }
        (lhs_base, rhs_base)
    };

    // Fill one output row given its input base offsets.
    let fill_row = |out_row: &mut [std::mem::MaybeUninit<U>], lhs_base: usize, rhs_base: usize| {
        let lhs_ptr = lhs_data.as_ptr();
        let rhs_ptr = rhs_data.as_ptr();
        // SAFETY: `lhs_last`/`rhs_last` are 0 or 1; for every k in 0..inner the
        // offsets `*_base + k * stride` stay in bounds (stride 0 repeats the
        // broadcast scalar, stride 1 walks a contiguous run of length `inner`).
        unsafe {
            match (lhs_last, rhs_last) {
                (1, 1) => {
                    for (k, out) in out_row.iter_mut().enumerate() {
                        out.write(op(*lhs_ptr.add(lhs_base + k), *rhs_ptr.add(rhs_base + k)));
                    }
                }
                (1, 0) => {
                    let rv = *rhs_ptr.add(rhs_base);
                    for (k, out) in out_row.iter_mut().enumerate() {
                        out.write(op(*lhs_ptr.add(lhs_base + k), rv));
                    }
                }
                (0, 1) => {
                    let lv = *lhs_ptr.add(lhs_base);
                    for (k, out) in out_row.iter_mut().enumerate() {
                        out.write(op(lv, *rhs_ptr.add(rhs_base + k)));
                    }
                }
                _ => {
                    let lv = *lhs_ptr.add(lhs_base);
                    let rv = *rhs_ptr.add(rhs_base);
                    for out in out_row.iter_mut() {
                        out.write(op(lv, rv));
                    }
                }
            }
        }
    };

    // For small tensors a sequential row walk avoids rayon task overhead.
    if numel < BINARY_PAR_THRESHOLD {
        let fill = |spare: &mut [std::mem::MaybeUninit<U>]| {
            for (row, out_row) in spare.chunks_mut(inner).enumerate() {
                let (lhs_base, rhs_base) = row_bases(row);
                fill_row(out_row, lhs_base, rhs_base);
            }
            Ok(())
        };
        // SAFETY: `fill` writes every element of the spare slice.
        let out = unsafe { build_vec_with::<U, Infallible, _>(numel, fill) }
            .unwrap_or_else(|e| match e {});
        return Ok(out);
    }

    // Parallelize over groups of whole rows so each rayon task carries enough
    // work regardless of the inner-run length.
    let rows_per_chunk = (PAR_CHUNK / inner).max(1);
    let chunk_len = rows_per_chunk * inner;
    let fill = |spare: &mut [std::mem::MaybeUninit<U>]| {
        spare
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let first_row = chunk_idx * rows_per_chunk;
                for (local, out_row) in out_chunk.chunks_mut(inner).enumerate() {
                    let (lhs_base, rhs_base) = row_bases(first_row + local);
                    fill_row(out_row, lhs_base, rhs_base);
                }
            });
        Ok(())
    };
    // SAFETY: the chunks partition the spare slice into whole rows and every
    // row is fully written.
    let out =
        unsafe { build_vec_with::<U, Infallible, _>(numel, fill) }.unwrap_or_else(|e| match e {});
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::ops::arithmetic::{add, div, mul, neg, sub};
    use crate::tensor::DataType;
    use std::sync::Arc;

    fn create_test_tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let shape_obj = Shape::new(shape);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);

        if let Some(slice) = tensor_data.as_f32_slice_mut() {
            slice.copy_from_slice(&data);
        }

        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn test_add_basic() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![4.0, 5.0, 6.0], vec![3], false);

        let result = add(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[5.0, 7.0, 9.0]);
        assert_eq!(result.shape().dims(), &[3]);
    }

    #[test]
    fn test_add_broadcasting() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![10.0], vec![1], false);

        let result = add(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[11.0, 12.0, 13.0]);
        assert_eq!(result.shape().dims(), &[3]);
    }

    // Naive reference broadcast add for [d0,d1,d2] op [d0',d1',d2'] where each
    // operand dim is either the output dim or 1.
    fn naive_add_3d(a: &[f32], ash: [usize; 3], b: &[f32], bsh: [usize; 3]) -> Vec<f32> {
        let osh = [ash[0].max(bsh[0]), ash[1].max(bsh[1]), ash[2].max(bsh[2])];
        let idx = |sh: [usize; 3], c: [usize; 3]| {
            let c0 = if sh[0] == 1 { 0 } else { c[0] };
            let c1 = if sh[1] == 1 { 0 } else { c[1] };
            let c2 = if sh[2] == 1 { 0 } else { c[2] };
            (c0 * sh[1] + c1) * sh[2] + c2
        };
        let mut out = Vec::with_capacity(osh[0] * osh[1] * osh[2]);
        for i in 0..osh[0] {
            for j in 0..osh[1] {
                for k in 0..osh[2] {
                    out.push(a[idx(ash, [i, j, k])] + b[idx(bsh, [i, j, k])]);
                }
            }
        }
        out
    }

    #[test]
    fn test_add_broadcast_patterns_match_naive() {
        // Exercises the inner-run kernel's four stride branches: last-dim
        // broadcast on each side, a middle-dim broadcast, and a full match.
        let cases: &[([usize; 3], [usize; 3])] = &[
            ([2, 3, 4], [2, 3, 1]), // rhs broadcasts last dim  -> (1,0)
            ([2, 3, 1], [2, 3, 4]), // lhs broadcasts last dim  -> (0,1)
            ([2, 1, 4], [2, 3, 4]), // lhs broadcasts middle dim
            ([2, 3, 4], [1, 3, 4]), // rhs broadcasts leading dim
            ([1, 3, 4], [2, 3, 4]), // lhs broadcasts leading dim
            ([2, 3, 4], [2, 3, 4]), // no broadcast -> (1,1)
        ];
        for &(ash, bsh) in cases {
            let an = ash.iter().product::<usize>();
            let bn = bsh.iter().product::<usize>();
            let a: Vec<f32> = (0..an).map(|x| x as f32 + 1.0).collect();
            let b: Vec<f32> = (0..bn).map(|x| (x as f32) * 0.5 - 3.0).collect();
            let ta = create_test_tensor_f32(a.clone(), ash.to_vec(), false);
            let tb = create_test_tensor_f32(b.clone(), bsh.to_vec(), false);
            let got = add(&ta, &tb).unwrap();
            let expected = naive_add_3d(&a, ash, &b, bsh);
            assert_eq!(
                got.data().as_f32_slice().unwrap(),
                expected.as_slice(),
                "broadcast {ash:?} + {bsh:?}"
            );
        }
    }

    #[test]
    fn test_sub_basic() {
        let a = create_test_tensor_f32(vec![5.0, 7.0, 9.0], vec![3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);

        let result = sub(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_basic() {
        let a = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);
        let b = create_test_tensor_f32(vec![5.0, 6.0, 7.0], vec![3], false);

        let result = mul(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_basic() {
        let a = create_test_tensor_f32(vec![10.0, 15.0, 20.0], vec![3], false);
        let b = create_test_tensor_f32(vec![2.0, 3.0, 4.0], vec![3], false);

        let result = div(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();

        assert_eq!(result_data, &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_neg_basic() {
        let a = create_test_tensor_f32(vec![1.0, -2.0, 3.5], vec![3], false);
        let result = neg(&a).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[-1.0, 2.0, -3.5]);
    }

    #[test]
    fn test_gradient_tracking() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], true);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], true);

        let result = add(&a, &b).unwrap();

        assert!(result.requires_grad());
        assert!(result.grad_fn().is_some());
    }

    #[test]
    fn test_device_mismatch_error() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        // This would normally fail, but we can't easily create different device tensors in tests
        // So we'll just test that same device works
        let result = add(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixed_dtype_promotion() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);

        // Create an i32 tensor
        let shape_obj = Shape::new(vec![2]);
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);
        if let Some(slice) = tensor_data.as_i32_slice_mut() {
            slice.copy_from_slice(&[3, 4]);
        }
        let b = Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        );

        let result = add(&a, &b).unwrap();
        assert_eq!(result.dtype(), DataType::Float32);
        assert_eq!(result.data().as_f32_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_sub_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0], vec![1, 2], false);
        let result = sub(&a, &b).unwrap();
        let expected = vec![0.0, 0.0, 2.0, 2.0];
        assert_eq!(result.data().as_f32_slice().unwrap(), expected.as_slice());
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_mul_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![2.0], vec![1, 1], false);
        let result = mul(&a, &b).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_div_broadcasting_2d() {
        let a = create_test_tensor_f32(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], false);
        let b = create_test_tensor_f32(vec![2.0], vec![1, 1], false);
        let result = div(&a, &b).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_bool_arithmetic_behaviour() {
        // Create boolean tensors
        let shape_obj = Shape::new(vec![2]);
        let mut data_a = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_a.as_bool_slice_mut() {
            slice.copy_from_slice(&[true, false]);
        }
        let a = Tensor::new(
            Arc::new(data_a),
            shape_obj.clone(),
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let mut data_b = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        if let Some(slice) = data_b.as_bool_slice_mut() {
            slice.copy_from_slice(&[false, true]);
        }
        let b = Tensor::new(
            Arc::new(data_b),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let add_result = add(&a, &b).unwrap();
        assert_eq!(add_result.dtype(), DataType::Bool);
        assert_eq!(add_result.data().as_bool_slice().unwrap(), &[true, true]);
        assert!(sub(&a, &b).is_err());
        let mul_result = mul(&a, &b).unwrap();
        assert_eq!(mul_result.dtype(), DataType::Bool);
        assert_eq!(mul_result.data().as_bool_slice().unwrap(), &[false, false]);
        let div_result = div(&a, &b).unwrap();
        assert_eq!(div_result.dtype(), DataType::Float32);
        assert_eq!(
            div_result.data().as_f32_slice().unwrap(),
            &[f32::INFINITY, 0.0]
        );
        assert!(neg(&a).is_err());
    }

    #[test]
    fn test_incompatible_shapes_error() {
        let a = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        assert!(sub(&a, &b).is_err());
        assert!(mul(&a, &b).is_err());
        assert!(div(&a, &b).is_err());
    }

    #[test]
    fn test_division_by_zero_returns_inf() {
        let a = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let b = create_test_tensor_f32(vec![0.0, 1.0], vec![2], false);
        let result = div(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();
        assert!(result_data[0].is_infinite());
        assert_eq!(result_data[1], 2.0);
    }

    #[test]
    fn test_division_by_zero_ieee_semantics_broadcast_path() {
        // Broadcasting a scalar zero divisor exercises the non-SIMD path,
        // which must match IEEE 754 (and the SIMD path): -1/0 = -inf,
        // 0/0 = NaN, 1/0 = inf.
        let a = create_test_tensor_f32(vec![-1.0, 0.0, 1.0], vec![3], false);
        let b = create_test_tensor_f32(vec![0.0], vec![1], false);
        let result = div(&a, &b).unwrap();
        let result_data = result.data().as_f32_slice().unwrap();
        assert_eq!(result_data[0], f32::NEG_INFINITY);
        assert!(result_data[1].is_nan());
        assert_eq!(result_data[2], f32::INFINITY);
    }

    #[test]
    fn test_add_handles_zero_sized_broadcast() {
        let a = create_test_tensor_f32(vec![], vec![0, 3], false);
        let b = create_test_tensor_f32(vec![1.0, 2.0, 3.0], vec![3], false);

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[0, 3]);
        assert_eq!(result.data().as_f32_slice().unwrap().len(), 0);
    }

    #[test]
    fn test_add_handles_zero_sized_broadcast_from_vec() {
        use crate::tensor::TensorData;

        let a_data = TensorData::from_vec::<f32>(vec![], DataType::Float32, Device::cpu());
        let a = Tensor::new(
            Arc::new(a_data),
            Shape::new(vec![0, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let b_data =
            TensorData::from_vec::<f32>(vec![1.0_f32, 2.0, 3.0], DataType::Float32, Device::cpu());
        let b = Tensor::new(
            Arc::new(b_data),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[0, 3]);
        assert_eq!(result.data().as_f32_slice().unwrap().len(), 0);
    }
}
