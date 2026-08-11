// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::{Tensor, TensorData},
};
use rayon::prelude::*;

/// Inclusive scan of `input` into `output` along one dimension.
///
/// A row-major tensor factors into `outer` independent slabs of
/// `dim_size * inner` elements; within a slab, position `d` along the scanned
/// dimension owns the contiguous run `[d * inner, (d + 1) * inner)`. Scanning
/// therefore reduces to `dim_size` contiguous vector steps per slab, and the
/// slabs parallelise cleanly because they are disjoint.
///
/// This replaces three hand-rolled variants per operation (1-D, 2-D, N-D) that
/// walked the scanned dimension with stride `inner` — one rayon task per
/// output *column*, writing through a pointer laundered as `usize` to cross
/// the closure boundary. The single-element strided writes were also the
/// worst possible access pattern for the leading-dimension case.
///
/// `reverse` scans from the last index toward the first (the `cumsum`
/// backward pass).
fn scan_along_dim<T, F>(
    output: &mut [T],
    input: &[T],
    dim_size: usize,
    inner: usize,
    reverse: bool,
    combine: F,
) where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let slab = dim_size * inner;
    if slab == 0 {
        return;
    }
    debug_assert_eq!(output.len(), input.len());
    debug_assert_eq!(output.len() % slab, 0);

    let run = |out: &mut [T], inp: &[T]| {
        if reverse {
            let last = (dim_size - 1) * inner;
            out[last..].copy_from_slice(&inp[last..]);
            for d in (0..dim_size.saturating_sub(1)).rev() {
                let (cur, next) = out[d * inner..(d + 2) * inner].split_at_mut(inner);
                for i in 0..inner {
                    cur[i] = combine(next[i], inp[d * inner + i]);
                }
            }
        } else {
            out[..inner].copy_from_slice(&inp[..inner]);
            for d in 1..dim_size {
                let (prev, cur) = out[(d - 1) * inner..(d + 1) * inner].split_at_mut(inner);
                for i in 0..inner {
                    cur[i] = combine(prev[i], inp[d * inner + i]);
                }
            }
        }
    };

    if output.len() == slab {
        run(output, input);
    } else {
        output
            .par_chunks_mut(slab)
            .zip(input.par_chunks(slab))
            .for_each(|(out_slab, in_slab)| run(out_slab, in_slab));
    }
}

macro_rules! cumprod_forward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if dim >= tensor.ndim() {
                return Err(MinitensorError::dim_out_of_range(
                    dim as isize,
                    tensor.ndim(),
                ));
            }

            let dim_size = shape[dim];
            let inner = shape[dim + 1..].iter().product::<usize>();
            scan_along_dim(
                output,
                input_data,
                dim_size,
                inner,
                false,
                |acc: $t, v: $t| acc * v,
            );
            Ok(())
        }
    };
}

/// Gradient of `cumprod` along a dimension.
///
/// With no zero in the run, `d/dx_i` is the running sum of `grad_j * y_j`
/// divided by `x_i`. A zero breaks that division, so the run is split at the
/// first zero `z`:
///
/// * before `z`, only outputs before `z` contribute -- every later one carries
///   the factor `x_z = 0` -- so the same running sum works, stopped at `z`;
/// * at `z` itself, the term is assembled from the prefix and a running suffix
///   product instead of by dividing;
/// * after `z`, every term carries `x_z` and the gradient is zero.
///
/// None of that reasoning depends on `z` being the *only* zero. A second zero
/// at `z2` simply drives the suffix product to zero once it is reached, so the
/// terms beyond it drop out on their own. There used to be a separate branch
/// for two or more zeros that zeroed the whole run, which discarded the
/// gradients before and at the first zero: `cumprod` of `[2, 0, 3, 0]` reported
/// `[0, 0, 0, 0]` where the true gradient is `[1, 8, 0, 0]`.
macro_rules! cumprod_backward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        pub(crate) fn $name(
            input: &Tensor,
            output: &Tensor,
            grad: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = input
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let out_data = output
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let grad_data = grad
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = input.shape().dims();

            if input.ndim() == 1 {
                if dim != 0 {
                    return Err(MinitensorError::dim_out_of_range(
                        dim as isize,
                        input.ndim(),
                    ));
                }
                let len = input_data.len();
                // count zeros and index
                let mut zero_count = 0;
                let mut zero_idx = 0;
                for i in 0..len {
                    if input_data[i] == 0 as $t {
                        zero_count += 1;
                        if zero_count == 1 {
                            zero_idx = i;
                        }
                    }
                }
                if zero_count == 0 {
                    let mut s: $t = 0 as $t;
                    for i in (0..len).rev() {
                        s += grad_data[i] * out_data[i];
                        output[i] = s / input_data[i];
                    }
                } else {
                    let mut s: $t = 0 as $t;
                    for i in (0..zero_idx).rev() {
                        s += grad_data[i] * out_data[i];
                        output[i] = s / input_data[i];
                    }
                    let mut prefix: $t = 1 as $t;
                    for i in 0..zero_idx {
                        prefix *= input_data[i];
                    }
                    let mut prod_suffix: $t = 1 as $t;
                    let mut grad_zero: $t = 0 as $t;
                    for j in zero_idx..len {
                        grad_zero += grad_data[j] * prod_suffix;
                        if j + 1 < len {
                            prod_suffix *= input_data[j + 1];
                        }
                    }
                    output[zero_idx] = grad_zero * prefix;
                    for i in zero_idx + 1..len {
                        output[i] = 0 as $t;
                    }
                }
            } else if input.ndim() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        for c in 0..cols {
                            let mut zero_count = 0;
                            let mut zero_idx = 0;
                            for r in 0..rows {
                                let idx = r * cols + c;
                                if input_data[idx] == 0 as $t {
                                    zero_count += 1;
                                    if zero_count == 1 {
                                        zero_idx = r;
                                    }
                                }
                            }
                            if zero_count == 0 {
                                let mut s: $t = 0 as $t;
                                for r in (0..rows).rev() {
                                    let idx = r * cols + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                            } else {
                                let mut s: $t = 0 as $t;
                                for r in (0..zero_idx).rev() {
                                    let idx = r * cols + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                                let mut prefix: $t = 1 as $t;
                                for r in 0..zero_idx {
                                    prefix *= input_data[r * cols + c];
                                }
                                let mut prod_suffix: $t = 1 as $t;
                                let mut grad_zero: $t = 0 as $t;
                                for r in zero_idx..rows {
                                    let idx = r * cols + c;
                                    grad_zero += grad_data[idx] * prod_suffix;
                                    if r + 1 < rows {
                                        prod_suffix *= input_data[(r + 1) * cols + c];
                                    }
                                }
                                let zero_index = zero_idx * cols + c;
                                output[zero_index] = grad_zero * prefix;
                                for r in zero_idx + 1..rows {
                                    let idx = r * cols + c;
                                    output[idx] = 0 as $t;
                                }
                            }
                        }
                    }
                    1 => {
                        for r in 0..rows {
                            let base = r * cols;
                            let mut zero_count = 0;
                            let mut zero_idx = 0;
                            for c in 0..cols {
                                let idx = base + c;
                                if input_data[idx] == 0 as $t {
                                    zero_count += 1;
                                    if zero_count == 1 {
                                        zero_idx = c;
                                    }
                                }
                            }
                            if zero_count == 0 {
                                let mut s: $t = 0 as $t;
                                for c in (0..cols).rev() {
                                    let idx = base + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                            } else {
                                let mut s: $t = 0 as $t;
                                for c in (0..zero_idx).rev() {
                                    let idx = base + c;
                                    s += grad_data[idx] * out_data[idx];
                                    output[idx] = s / input_data[idx];
                                }
                                let mut prefix: $t = 1 as $t;
                                for c in 0..zero_idx {
                                    prefix *= input_data[base + c];
                                }
                                let mut prod_suffix: $t = 1 as $t;
                                let mut grad_zero: $t = 0 as $t;
                                for c in zero_idx..cols {
                                    let idx = base + c;
                                    grad_zero += grad_data[idx] * prod_suffix;
                                    if c + 1 < cols {
                                        prod_suffix *= input_data[base + c + 1];
                                    }
                                }
                                output[base + zero_idx] = grad_zero * prefix;
                                for c in zero_idx + 1..cols {
                                    output[base + c] = 0 as $t;
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(MinitensorError::dim_out_of_range(
                            dim as isize,
                            input.ndim(),
                        ));
                    }
                }
            } else {
                let dim_size = shape[dim];
                let inner = shape[dim + 1..].iter().product::<usize>();
                let outer = shape[..dim].iter().product::<usize>();
                let total = outer * inner;
                for idx in 0..total {
                    let o = idx / inner;
                    let r = idx % inner;
                    let base = o * dim_size * inner + r;
                    let mut zero_count = 0;
                    let mut zero_idx = 0;
                    for d in 0..dim_size {
                        let i = base + d * inner;
                        if input_data[i] == 0 as $t {
                            zero_count += 1;
                            if zero_count == 1 {
                                zero_idx = d;
                            }
                        }
                    }
                    if zero_count == 0 {
                        let mut s: $t = 0 as $t;
                        for d in (0..dim_size).rev() {
                            let i = base + d * inner;
                            s += grad_data[i] * out_data[i];
                            output[i] = s / input_data[i];
                        }
                    } else {
                        let mut s: $t = 0 as $t;
                        for d in (0..zero_idx).rev() {
                            let i = base + d * inner;
                            s += grad_data[i] * out_data[i];
                            output[i] = s / input_data[i];
                        }
                        let mut prefix: $t = 1 as $t;
                        for d in 0..zero_idx {
                            prefix *= input_data[base + d * inner];
                        }
                        let mut prod_suffix: $t = 1 as $t;
                        let mut grad_zero: $t = 0 as $t;
                        for d in zero_idx..dim_size {
                            let i = base + d * inner;
                            grad_zero += grad_data[i] * prod_suffix;
                            if d + 1 < dim_size {
                                prod_suffix *= input_data[base + (d + 1) * inner];
                            }
                        }
                        let zero_index = base + zero_idx * inner;
                        output[zero_index] = grad_zero * prefix;
                        for d in zero_idx + 1..dim_size {
                            output[base + d * inner] = 0 as $t;
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

macro_rules! cumsum_forward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if dim >= tensor.ndim() {
                return Err(MinitensorError::dim_out_of_range(
                    dim as isize,
                    tensor.ndim(),
                ));
            }

            let dim_size = shape[dim];
            let inner = shape[dim + 1..].iter().product::<usize>();
            scan_along_dim(
                output,
                input_data,
                dim_size,
                inner,
                false,
                |acc: $t, v: $t| acc + v,
            );
            Ok(())
        }
    };
}

macro_rules! cumsum_backward {
    ($name:ident, $get:ident, $get_mut:ident, $t:ty) => {
        pub(crate) fn $name(
            tensor: &Tensor,
            result_data: &mut TensorData,
            dim: usize,
        ) -> Result<()> {
            let input_data = tensor
                .data()
                .$get()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get slice"))?;
            let output = result_data
                .$get_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable slice"))?;
            let shape = tensor.shape().dims();

            if dim >= tensor.ndim() {
                return Err(MinitensorError::dim_out_of_range(
                    dim as isize,
                    tensor.ndim(),
                ));
            }

            let dim_size = shape[dim];
            let inner = shape[dim + 1..].iter().product::<usize>();
            scan_along_dim(
                output,
                input_data,
                dim_size,
                inner,
                true,
                |acc: $t, v: $t| acc + v,
            );
            Ok(())
        }
    };
}

cumprod_forward!(cumprod_f32, as_f32_slice, as_f32_slice_mut, f32);
cumprod_forward!(cumprod_f64, as_f64_slice, as_f64_slice_mut, f64);
cumprod_forward!(cumprod_i32, as_i32_slice, as_i32_slice_mut, i32);
cumprod_forward!(cumprod_i64, as_i64_slice, as_i64_slice_mut, i64);

cumprod_backward!(cumprod_backward_f32, as_f32_slice, as_f32_slice_mut, f32);
cumprod_backward!(cumprod_backward_f64, as_f64_slice, as_f64_slice_mut, f64);

cumsum_forward!(cumsum_f32, as_f32_slice, as_f32_slice_mut, f32);
cumsum_forward!(cumsum_f64, as_f64_slice, as_f64_slice_mut, f64);
cumsum_forward!(cumsum_i32, as_i32_slice, as_i32_slice_mut, i32);
cumsum_forward!(cumsum_i64, as_i64_slice, as_i64_slice_mut, i64);

cumsum_backward!(cumsum_backward_f32, as_f32_slice, as_f32_slice_mut, f32);
cumsum_backward!(cumsum_backward_f64, as_f64_slice, as_f64_slice_mut, f64);
cumsum_backward!(cumsum_backward_i32, as_i32_slice, as_i32_slice_mut, i32);
cumsum_backward!(cumsum_backward_i64, as_i64_slice, as_i64_slice_mut, i64);

#[cfg(test)]
mod tests {
    use super::*;
    // Sibling reduction entry points under test; the non-test build of this
    // module does not reference them.
    use crate::device::Device;
    use crate::ops::reduction::*;
    use crate::tensor::{DataType, Shape};
    use std::sync::Arc;

    fn create_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);
        tensor_data
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn create_tensor_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Int32);
        tensor_data
            .as_i32_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn create_tensor_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Bool);
        tensor_data
            .as_bool_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        Tensor::new(
            Arc::new(tensor_data),
            shape_obj,
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    #[test]
    fn test_median_global_even_length() {
        let t = create_tensor_f32(vec![3.0, 1.0, 4.0, 2.0], vec![4]);
        let (value, indices) = median(&t, None, false).unwrap();
        assert!(indices.is_none());
        assert!(value.shape().is_scalar());
        let result = value.data().as_f32_slice().unwrap();
        assert_eq!(result, &[2.0]);
    }

    #[test]
    fn test_median_with_dim_returns_indices() {
        let t = create_tensor_f32(vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0], vec![2, 3]);
        let (values, indices_opt) = median(&t, Some(1), false).unwrap();
        let indices = indices_opt.unwrap();
        assert_eq!(values.shape().dims(), &[2]);
        assert_eq!(indices.shape().dims(), &[2]);
        let values_slice = values.data().as_f32_slice().unwrap();
        let indices_slice = indices.data().as_i64_slice().unwrap();
        assert_eq!(values_slice, &[2.0, 5.0]);
        assert_eq!(indices_slice, &[2, 2]);
    }

    #[test]
    fn test_median_keepdim_preserves_rank() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let (values, indices_opt) = median(&t, Some(1), true).unwrap();
        let indices = indices_opt.unwrap();
        assert_eq!(values.shape().dims(), &[2, 1]);
        assert_eq!(indices.shape().dims(), &[2, 1]);
        assert_eq!(values.data().as_f32_slice().unwrap(), &[1.0, 3.0]);
        assert_eq!(indices.data().as_i64_slice().unwrap(), &[0, 0]);
    }

    #[test]
    fn test_median_empty_tensor_errors() {
        let t = create_tensor_f32(vec![], vec![0]);
        assert!(median(&t, None, false).is_err());
    }

    #[test]
    fn test_quantiles_all_multiple_probs() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = quantiles(
            &t,
            &[0.25, 0.75],
            None,
            false,
            QuantileInterpolation::Linear,
        )
        .unwrap();
        assert_eq!(result.shape().dims(), &[2]);
        let values = result.data().as_f32_slice().unwrap();
        assert!((values[0] - 1.75).abs() < 1e-6);
        assert!((values[1] - 3.25).abs() < 1e-6);
    }

    #[test]
    fn test_quantiles_dim_keepdim_layout() {
        let t = create_tensor_f32(vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0], vec![2, 3]);
        let result = quantiles(
            &t,
            &[0.5, 0.9],
            Some(1),
            true,
            QuantileInterpolation::Linear,
        )
        .unwrap();
        assert_eq!(result.shape().dims(), &[2, 2, 1]);
        let values = result.data().as_f32_slice().unwrap();
        let expected = [2.0, 5.0, 2.8, 5.8];
        for (value, target) in values.iter().zip(expected.iter()) {
            assert!((*value - *target).abs() < 1e-6);
        }
    }

    #[test]
    fn test_argmax_along_dim() {
        let t = create_tensor_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let result = argmax(&t, Some(1), false).unwrap();
        let res = result.data().as_i64_slice().unwrap();
        assert_eq!(res, &[1, 2]);
    }

    #[test]
    fn test_argmin_along_dim_keepdim() {
        let t = create_tensor_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let result = argmin(&t, Some(1), true).unwrap();
        assert_eq!(result.shape().dims(), &[2, 1]);
        let res = result.data().as_i64_slice().unwrap();
        assert_eq!(res, &[0, 1]);
    }

    #[test]
    fn test_all_any_global() {
        let t = create_tensor_i32(vec![1, 0, 2, 3], vec![2, 2]);
        let all_res = all(&t, None, false).unwrap();
        let any_res = any(&t, None, false).unwrap();
        assert!(!all_res.data().as_bool_slice().unwrap()[0]);
        assert!(any_res.data().as_bool_slice().unwrap()[0]);
    }

    #[test]
    fn test_all_along_dim() {
        let t = create_tensor_bool(vec![true, false, true, true], vec![2, 2]);
        let res = all(&t, Some(1), false).unwrap();
        assert_eq!(res.data().as_bool_slice().unwrap(), &[false, true]);
    }

    #[test]
    fn test_sum_global_and_keepdim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum(&t, None, false).unwrap();
        assert_eq!(s.shape().dims(), &[] as &[usize]);
        assert_eq!(s.data().as_f32_slice().unwrap()[0], 10.0);
        let s_keep = sum(&t, None, true).unwrap();
        assert_eq!(s_keep.shape().dims(), &[1, 1]);
        assert_eq!(s_keep.data().as_f32_slice().unwrap()[0], 10.0);
    }

    #[test]
    fn test_topk_largest_float() {
        let t = create_tensor_f32(vec![1.0, 3.0, 2.0, 4.0, -1.0, 5.0], vec![2, 3]);
        let (values, indices) = topk(&t, 2, Some(1), true, true).unwrap();
        assert_eq!(values.shape().dims(), &[2, 2]);
        assert_eq!(indices.shape().dims(), &[2, 2]);
        let values_slice = values.data().as_f32_slice().unwrap();
        let indices_slice = indices.data().as_i64_slice().unwrap();
        assert_eq!(values_slice, &[3.0, 2.0, 5.0, 4.0]);
        assert_eq!(indices_slice, &[1, 2, 2, 0]);
    }

    #[test]
    fn test_topk_smallest_unsorted() {
        let t = create_tensor_f32(vec![1.0, -2.0, 3.5, 0.0], vec![4]);
        let (values, indices) = topk(&t, 2, None, false, false).unwrap();
        assert_eq!(values.shape().dims(), &[2]);
        let mut pairs: Vec<(i64, f32)> = indices
            .data()
            .as_i64_slice()
            .unwrap()
            .iter()
            .zip(values.data().as_f32_slice().unwrap())
            .map(|(&i, &v)| (i, v))
            .collect();
        pairs.sort_by_key(|p| p.0);
        assert_eq!(pairs, vec![(1, -2.0), (3, 0.0)]);
    }

    #[test]
    fn test_topk_sorted_partial_ties_by_first_index() {
        let t = create_tensor_f32(vec![1.0, 5.0, 5.0, 4.0, 3.0, 2.0], vec![6]);
        let (values, indices) = topk(&t, 3, None, true, true).unwrap();

        assert_eq!(values.data().as_f32_slice().unwrap(), &[5.0, 5.0, 4.0]);
        assert_eq!(indices.data().as_i64_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_topk_sorted_smallest_bool() {
        let t = create_tensor_bool(vec![true, false, true, false], vec![4]);
        let (values, indices) = topk(&t, 2, None, false, true).unwrap();

        assert_eq!(values.data().as_bool_slice().unwrap(), &[false, false]);
        assert_eq!(indices.data().as_i64_slice().unwrap(), &[1, 3]);
    }

    #[test]
    fn test_topk_sorted_partial_nan_ordering() {
        let t = create_tensor_f32(vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0], vec![5]);
        let (values, indices) = topk(&t, 3, None, true, true).unwrap();
        let values = values.data().as_f32_slice().unwrap();

        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
        assert_eq!(indices.data().as_i64_slice().unwrap(), &[1, 3, 2]);
    }

    #[test]
    fn test_sum_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = sum(&t, Some(vec![0]), false).unwrap();
        assert_eq!(res.shape().dims(), &[2]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_sum_bool_counts_into_int64() {
        // Summing a mask counts its true entries, in `int64`. `bool` itself
        // has nothing to accumulate in.
        let t = create_tensor_bool(vec![true, false, true, true], vec![2, 2]);

        let per_column = sum(&t, Some(vec![0]), false).unwrap();
        assert_eq!(per_column.dtype(), DataType::Int64);
        assert_eq!(per_column.data().as_i64_slice().unwrap(), &[2, 1]);

        let total = sum(&t, None, false).unwrap();
        assert_eq!(total.dtype(), DataType::Int64);
        assert_eq!(total.data().as_i64_slice().unwrap(), &[3]);
    }

    #[test]
    fn test_sum_multi_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = sum(&t, Some(vec![0, 1]), false).unwrap();
        assert!(res.shape().is_scalar());
        assert_eq!(res.data().as_f32_slice().unwrap()[0], 10.0);
        let res_keep = sum(&t, Some(vec![0, 1]), true).unwrap();
        assert_eq!(res_keep.shape().dims(), &[1, 1]);
        assert_eq!(res_keep.data().as_f32_slice().unwrap()[0], 10.0);
    }

    #[test]
    fn test_prod_global_and_keepdim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let p = prod(&t, None, false).unwrap();
        assert_eq!(p.data().as_f32_slice().unwrap()[0], 24.0);
        let p_keep = prod(&t, None, true).unwrap();
        assert_eq!(p_keep.shape().dims(), &[1, 1]);
        assert_eq!(p_keep.data().as_f32_slice().unwrap()[0], 24.0);
    }

    #[test]
    fn test_prod_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = prod(&t, Some(vec![0]), false).unwrap();
        assert_eq!(res.data().as_f32_slice().unwrap(), &[3.0, 8.0]);
    }

    #[test]
    fn test_mean_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = mean(&t, Some(vec![1]), true).unwrap();
        assert_eq!(res.shape().dims(), &[2, 1]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_int_support() {
        let t = create_tensor_i32(vec![1, 2, 3, 4], vec![2, 2]);
        let res = mean(&t, Some(vec![0isize]), false).unwrap();
        assert_eq!(res.dtype(), DataType::Float32);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[2.0, 3.0]);
    }
}
