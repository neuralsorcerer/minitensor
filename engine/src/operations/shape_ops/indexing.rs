// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn expand_repeats(spec: RepeatInterleaveSpec<'_>, dim_size: usize) -> Result<Vec<usize>> {
    match spec {
        RepeatInterleaveSpec::Scalar(value) => Ok(vec![value; dim_size]),
        RepeatInterleaveSpec::Slice(values) => {
            if values.len() == dim_size {
                Ok(values.to_vec())
            } else if values.len() == 1 {
                if dim_size == 0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![values[0]; dim_size])
                }
            } else if values.is_empty() && dim_size == 0 {
                Ok(Vec::new())
            } else {
                Err(MinitensorError::invalid_operation(
                    "repeat_interleave: repeats must be a single value or match tensor size along dim"
                        .to_string(),
                ))
            }
        }
        RepeatInterleaveSpec::Tensor(tensor) => collect_repeats_from_tensor(tensor, dim_size),
    }
}

fn build_empty_repeat_result(tensor: &Tensor, dim: usize, target: usize) -> Result<Tensor> {
    let mut out_shape = tensor.shape().dims().to_vec();
    out_shape[dim] = target;
    let shape = Shape::new(out_shape);
    let dtype = tensor.dtype();
    let device = tensor.device();
    let data = TensorData::zeros_on_device(shape.numel(), dtype, device);
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        dtype,
        device,
        tensor.requires_grad(),
    ))
}

/// Repeat elements of ``tensor`` according to ``repeats`` along ``dim``.
pub fn repeat_interleave(
    tensor: &Tensor,
    repeats: RepeatInterleaveSpec<'_>,
    dim: Option<isize>,
    output_size: Option<usize>,
) -> Result<Tensor> {
    if dim.is_none() {
        let flat = tensor.flatten_all()?;
        return repeat_interleave(&flat, repeats, Some(0), output_size);
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave currently supports only CPU tensors".to_string(),
        ));
    }

    let dim = normalize_dim(dim.unwrap(), tensor.ndim())?;
    let dims = tensor.shape().dims();
    let dim_size = dims[dim];
    let reps = expand_repeats(repeats, dim_size)?;
    let total_repeats: usize = reps.iter().sum();

    if let Some(expected) = output_size {
        if expected != total_repeats {
            return Err(MinitensorError::invalid_argument(format!(
                "repeat_interleave: output_size ({expected}) must equal sum of repeats ({total_repeats})"
            )));
        }
    }

    let dtype = tensor.dtype();
    let device = tensor.device();
    let requires_grad = tensor.requires_grad();

    let target_dim = output_size.unwrap_or(total_repeats);
    let mut output_shape = dims.to_vec();
    output_shape[dim] = target_dim;
    let output_shape_obj = Shape::new(output_shape);
    let output_numel = output_shape_obj.numel();

    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = if dim == 0 {
        1
    } else {
        dims[..dim].iter().product()
    };

    let build_grad_fn = |repeats: Vec<usize>| {
        Arc::new(RepeatInterleaveBackward {
            input_shape: dims.to_vec(),
            repeats,
            input_id: tensor.id(),
            dim,
        })
    };

    if target_dim == 0 || output_numel == 0 || inner == 0 || outer == 0 {
        let mut result = build_empty_repeat_result(tensor, dim, target_dim)?;
        if requires_grad {
            let grad_fn = build_grad_fn(reps);
            result.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&result, Some(grad_fn))?;
        }
        return Ok(result);
    }

    macro_rules! repeat_impl {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave: tensor data access failed".to_string(),
                )
            })?;
            let mut out = vec![<$ty>::default(); output_numel];
            out.par_chunks_mut(target_dim * inner).enumerate().for_each(
                |(outer_idx, out_chunk)| {
                    let mut dst_offset = 0;
                    let base = outer_idx * dim_size * inner;
                    for (i, &rep) in reps.iter().enumerate() {
                        if rep == 0 {
                            continue;
                        }
                        let src_start = base + i * inner;
                        let src_slice = &src[src_start..src_start + inner];
                        for _ in 0..rep {
                            let end = dst_offset + inner;
                            out_chunk[dst_offset..end].copy_from_slice(src_slice);
                            dst_offset = end;
                        }
                    }
                },
            );
            TensorData::$from_vec(out, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => repeat_impl!(f32, as_f32_slice, from_vec_f32),
        DataType::Float64 => repeat_impl!(f64, as_f64_slice, from_vec_f64),
        DataType::Int32 => repeat_impl!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_impl!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => repeat_impl!(bool, as_bool_slice, from_vec_bool),
    };

    let mut result = Tensor::new(
        Arc::new(data),
        output_shape_obj,
        dtype,
        device,
        requires_grad,
    );

    if requires_grad {
        let grad_fn = build_grad_fn(reps);
        result.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result, Some(grad_fn))?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, TensorData},
    };

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
    fn test_reshape_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let reshaped = reshape(&tensor, Shape::new(vec![3, 2])).unwrap();

        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);

        let data = reshaped.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_invalid_size() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = reshape(&tensor, Shape::new(vec![2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_infer_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], false);
        let reshaped = reshape_with_inference(&tensor, vec![2, -1]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_reshape_multiple_negative_one_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let result = reshape_with_inference(&tensor, vec![-1, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_infer_mismatch_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], false);
        let result = reshape_with_inference(&tensor, vec![4, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_zero_dim_with_inference_error() {
        let tensor = create_test_tensor_f32(vec![], vec![0], false);
        let result = reshape_with_inference(&tensor, vec![-1, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let s0 = squeeze(&tensor, Some(0)).unwrap();
        assert_eq!(s0.shape().dims(), &[4, 1]);

        let s1 = squeeze(&s0, Some(1)).unwrap();
        assert_eq!(s1.shape().dims(), &[4]);

        let s_neg = squeeze(&tensor, Some(-1)).unwrap();
        assert_eq!(s_neg.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], false);

        let squeezed = squeeze(&tensor, None).unwrap();
        assert_eq!(squeezed.shape().dims(), &[4]);

        let scalar = create_test_tensor_f32(vec![1.0], vec![1, 1], false);
        let s = squeeze(&scalar, None).unwrap();
        assert!(s.shape().dims().is_empty());
    }

    #[test]
    fn test_squeeze_out_of_range() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        assert!(squeeze(&tensor, Some(2)).is_err());
        assert!(squeeze(&tensor, Some(-3)).is_err());
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);

        let u0 = unsqueeze(&tensor, 0).unwrap();
        assert_eq!(u0.shape().dims(), &[1, 4]);

        let u1 = unsqueeze(&tensor, 1).unwrap();
        assert_eq!(u1.shape().dims(), &[4, 1]);

        let u_neg = unsqueeze(&tensor, -1).unwrap();
        assert_eq!(u_neg.shape().dims(), &[4, 1]);
    }

    #[test]
    fn test_gradient_tracking() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);

        let reshaped = reshape(&tensor, Shape::new(vec![4])).unwrap();

        assert!(reshaped.requires_grad());
        assert!(reshaped.grad_fn().is_some());
    }

    #[test]
    fn test_concatenate_validation() {
        let tensor1 = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let tensor2 = create_test_tensor_f32(vec![3.0, 4.0], vec![2], false);

        let result = concatenate(&[&tensor1, &tensor2], 0).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_index_select_validation() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = index_select(&tensor, 1, &[0, 2]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice_empty_range() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = slice(&tensor, 1, 1, 1, 1).unwrap();
        assert_eq!(result.shape().dims(), &[2, 0]);
        assert_eq!(result.numel(), 0);
    }

    #[test]
    fn test_slice_empty_at_end() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = slice(&tensor, 0, 2, 2, 1).unwrap();
        assert_eq!(result.shape().dims(), &[0, 2]);
        assert_eq!(result.numel(), 0);
    }

    #[test]
    fn test_index_select_empty_indices() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);

        let result = index_select(&tensor, 1, &[]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 0]);
        assert_eq!(result.numel(), 0);
    }

    #[test]
    fn test_slice_validation() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);

        let result = slice(&tensor, 1, 0, 2, 1).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_repeat_basic() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        let repeated = repeat(&tensor, &[3]).unwrap();
        assert_eq!(repeated.shape().dims(), &[6]);
        let data = repeated.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_zero_numel_shape() {
        let tensor = create_test_tensor_f32(vec![], vec![0, 2], false);
        let repeated = repeat(&tensor, &[2, 3]).unwrap();
        assert_eq!(repeated.shape().dims(), &[0, 6]);
        assert_eq!(repeated.numel(), 0);
    }

    #[test]
    fn test_repeat_dim_mismatch_error() {
        let tensor = create_test_tensor_f32(vec![1.0, 2.0], vec![2], false);
        assert!(repeat(&tensor, &[]).is_err());
    }
}
