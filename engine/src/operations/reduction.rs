// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{add_to_graph, SumBackward},
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

/// Sum reduction along specified dimensions
pub fn sum(tensor: &Tensor, dim: Option<Vec<usize>>, keepdim: bool) -> Result<Tensor> {
    let dims_clone = dim.clone();
    let result = match dim {
        None => {
            // Sum all elements
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => sum_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => sum_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => sum_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => sum_all_i64(tensor, &mut result_data)?,
                DataType::Bool => {
                    return Err(MinitensorError::invalid_operation(
                        "Sum not supported for boolean tensors",
                    ))
                }
            }

            Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            )
        }
        Some(dims) => {
            // Sum along specific dimensions
            if dims.is_empty() {
                tensor.clone()
            } else if dims.len() == 1 {
                sum_along_dim(tensor, dims[0], keepdim)?
            } else {
                return Err(MinitensorError::not_implemented(
                    "Multi-dimensional reduction not yet implemented",
                ));
            }
        }
    };

    if result.requires_grad() {
        let grad_fn = Arc::new(SumBackward {
            input_id: tensor.id(),
            input_shape: tensor.shape().dims().to_vec(),
            dims: dims_clone,
            keepdim,
        });
        let mut result_with_grad = result;
        result_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&result_with_grad, Some(grad_fn))?;
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}

/// Mean reduction along specified dimensions
pub fn mean(tensor: &Tensor, dim: Option<Vec<usize>>, keepdim: bool) -> Result<Tensor> {
    let sum_result = sum(tensor, dim.clone(), keepdim)?;

    // Compute the number of elements being averaged
    let num_elements = match dim {
        None => tensor.numel() as f64,
        Some(dims) => {
            if dims.is_empty() {
                return Ok(tensor.clone());
            }

            let mut count = 1.0;
            for &d in &dims {
                if d < tensor.ndim() {
                    count *= tensor.shape().dims()[d] as f64;
                }
            }
            count
        }
    };

    // Divide by number of elements
    let divisor = match tensor.dtype() {
        DataType::Float32 => Tensor::new(
            Arc::new(TensorData::from_vec(
                vec![num_elements as f32],
                DataType::Float32,
                tensor.device(),
            )),
            Shape::scalar(),
            DataType::Float32,
            tensor.device(),
            false,
        ),
        DataType::Float64 => Tensor::new(
            Arc::new(TensorData::from_vec(
                vec![num_elements],
                DataType::Float64,
                tensor.device(),
            )),
            Shape::scalar(),
            DataType::Float64,
            tensor.device(),
            false,
        ),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Mean only supported for floating point tensors",
            ))
        }
    };

    crate::operations::arithmetic::div(&sum_result, &divisor)
}

/// Logical all reduction along specified dimension
pub fn all(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => all_all(tensor, keepdim),
        Some(d) => all_along_dim(tensor, d, keepdim),
    }
}

/// Logical any reduction along specified dimension
pub fn any(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => any_all(tensor, keepdim),
        Some(d) => any_along_dim(tensor, d, keepdim),
    }
}

fn all_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut result_data = TensorData::zeros_on_device(1, DataType::Bool, tensor.device());
    let out_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let all_true = match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?
            .par_iter()
            .all(|&x| x != 0.0),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?
            .par_iter()
            .all(|&x| x != 0.0),
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?
            .par_iter()
            .all(|&x| x != 0),
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?
            .par_iter()
            .all(|&x| x != 0),
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?
            .par_iter()
            .all(|&x| x),
    };
    out_slice[0] = all_true;
    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn any_all(tensor: &Tensor, keepdim: bool) -> Result<Tensor> {
    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };
    let mut result_data = TensorData::zeros_on_device(1, DataType::Bool, tensor.device());
    let out_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
    let any_true = match tensor.dtype() {
        DataType::Float32 => tensor
            .data()
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?
            .par_iter()
            .any(|&x| x != 0.0),
        DataType::Float64 => tensor
            .data()
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?
            .par_iter()
            .any(|&x| x != 0.0),
        DataType::Int32 => tensor
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?
            .par_iter()
            .any(|&x| x != 0),
        DataType::Int64 => tensor
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?
            .par_iter()
            .any(|&x| x != 0),
        DataType::Bool => tensor
            .data()
            .as_bool_slice()
            .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?
            .par_iter()
            .any(|&x| x),
    };
    out_slice[0] = any_true;
    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn all_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape.clone());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Bool, tensor.device());

    let dim_size = input_shape[dim];
    let _outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0.0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0.0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] == 0 {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = true;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if !input[in_idx] {
                        val = false;
                        break;
                    }
                }
                *out = val;
            });
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

fn any_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape.clone());
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Bool, tensor.device());

    let dim_size = input_shape[dim];
    let _outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0.0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0.0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] != 0 {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;
            output.par_iter_mut().enumerate().for_each(|(idx, out)| {
                let o = idx / inner;
                let r = idx % inner;
                let mut val = false;
                for d in 0..dim_size {
                    let in_idx = o * outer_stride + d * inner + r;
                    if input[in_idx] {
                        val = true;
                        break;
                    }
                }
                *out = val;
            });
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// Maximum value along specified dimension
pub fn max(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global maximum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => max_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => max_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => max_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => max_all_i64(tensor, &mut result_data)?,
                DataType::Bool => max_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            ))
        }
        Some(d) => max_along_dim(tensor, d, keepdim),
    }
}

/// Minimum value along specified dimension
pub fn min(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global minimum
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

            match tensor.dtype() {
                DataType::Float32 => min_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => min_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => min_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => min_all_i64(tensor, &mut result_data)?,
                DataType::Bool => min_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                tensor.dtype(),
                tensor.device(),
                tensor.requires_grad(),
            ))
        }
        Some(d) => min_along_dim(tensor, d, keepdim),
    }
}

/// Argument of maximum value along specified dimension
pub fn argmax(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global argmax
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmax_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmax_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmax_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmax_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmax_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmax doesn't require gradients
            ))
        }
        Some(d) => argmax_along_dim(tensor, d, keepdim),
    }
}

/// Argument of minimum value along specified dimension
pub fn argmin(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    match dim {
        None => {
            // Find global argmin
            let result_shape = if keepdim {
                Shape::new(vec![1; tensor.ndim()])
            } else {
                Shape::scalar()
            };

            let mut result_data = TensorData::zeros_on_device(1, DataType::Int64, tensor.device());

            match tensor.dtype() {
                DataType::Float32 => argmin_all_f32(tensor, &mut result_data)?,
                DataType::Float64 => argmin_all_f64(tensor, &mut result_data)?,
                DataType::Int32 => argmin_all_i32(tensor, &mut result_data)?,
                DataType::Int64 => argmin_all_i64(tensor, &mut result_data)?,
                DataType::Bool => argmin_all_bool(tensor, &mut result_data)?,
            }

            Ok(Tensor::new(
                Arc::new(result_data),
                result_shape,
                DataType::Int64,
                tensor.device(),
                false, // argmin doesn't require gradients
            ))
        }
        Some(d) => argmin_along_dim(tensor, d, keepdim),
    }
}

/// Standard deviation along specified dimension
pub fn std(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    let variance = var(tensor, dim, keepdim)?;
    crate::operations::activation::sqrt(&variance)
}

/// Variance along specified dimension
pub fn var(tensor: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "Variance only supported for floating point tensors",
        ));
    }

    // Compute mean
    let mean_tensor = mean(tensor, dim.map(|d| vec![d]), keepdim)?;

    // Compute (x - mean)^2
    let diff = crate::operations::arithmetic::sub(tensor, &mean_tensor)?;
    let squared_diff = crate::operations::arithmetic::mul(&diff, &diff)?;

    // Compute mean of squared differences
    mean(&squared_diff, dim.map(|d| vec![d]), keepdim)
}

// Helper functions for type-specific operations

fn sum_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let sum: f32 = data.par_iter().sum();

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let sum: f64 = data.par_iter().sum();

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let sum: i32 = data.par_iter().sum();

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let sum: i64 = data.par_iter().sum();

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = sum;
    Ok(())
}

fn sum_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();

    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => sum_along_dim_f32(tensor, &mut result_data, dim)?,
        DataType::Float64 => sum_along_dim_f64(tensor, &mut result_data, dim)?,
        DataType::Int32 => sum_along_dim_i32(tensor, &mut result_data, dim)?,
        DataType::Int64 => sum_along_dim_i64(tensor, &mut result_data, dim)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Sum not supported for boolean tensors",
            ))
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn sum_along_dim_f32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    let input_shape = tensor.shape().dims();
    let _dim_size = input_shape[dim];

    // Simple implementation for 2D case
    if tensor.ndim() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        if dim == 0 {
            // Sum along rows (result has shape [cols])
            for j in 0..cols {
                let mut sum = 0.0;
                for i in 0..rows {
                    sum += input_data[i * cols + j];
                }
                result_slice[j] = sum;
            }
        } else {
            // Sum along columns (result has shape [rows])
            for i in 0..rows {
                let mut sum = 0.0;
                for j in 0..cols {
                    sum += input_data[i * cols + j];
                }
                result_slice[i] = sum;
            }
        }
    } else {
        // For higher dimensions, use a more general approach
        return Err(MinitensorError::not_implemented(
            "Sum along dimension for >2D tensors not yet implemented",
        ));
    }

    Ok(())
}

fn sum_along_dim_f64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    let input_shape = tensor.shape().dims();
    let _dim_size = input_shape[dim];

    // Simple implementation for 2D case
    if tensor.ndim() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        if dim == 0 {
            for j in 0..cols {
                let mut sum = 0.0;
                for i in 0..rows {
                    sum += input_data[i * cols + j];
                }
                result_slice[j] = sum;
            }
        } else {
            for i in 0..rows {
                let mut sum = 0.0;
                for j in 0..cols {
                    sum += input_data[i * cols + j];
                }
                result_slice[i] = sum;
            }
        }
    } else {
        return Err(MinitensorError::not_implemented(
            "Sum along dimension for >2D tensors not yet implemented",
        ));
    }

    Ok(())
}

fn sum_along_dim_i32(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    let input_shape = tensor.shape().dims();
    let _dim_size = input_shape[dim];

    if tensor.ndim() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        if dim == 0 {
            for j in 0..cols {
                let mut sum = 0;
                for i in 0..rows {
                    sum += input_data[i * cols + j];
                }
                result_slice[j] = sum;
            }
        } else {
            for i in 0..rows {
                let mut sum = 0;
                for j in 0..cols {
                    sum += input_data[i * cols + j];
                }
                result_slice[i] = sum;
            }
        }
    } else {
        return Err(MinitensorError::not_implemented(
            "Sum along dimension for >2D tensors not yet implemented",
        ));
    }

    Ok(())
}

fn sum_along_dim_i64(tensor: &Tensor, result_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    let input_shape = tensor.shape().dims();
    let _dim_size = input_shape[dim];

    if tensor.ndim() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        if dim == 0 {
            for j in 0..cols {
                let mut sum = 0;
                for i in 0..rows {
                    sum += input_data[i * cols + j];
                }
                result_slice[j] = sum;
            }
        } else {
            for i in 0..rows {
                let mut sum = 0;
                for j in 0..cols {
                    sum += input_data[i * cols + j];
                }
                result_slice[i] = sum;
            }
        }
    } else {
        return Err(MinitensorError::not_implemented(
            "Sum along dimension for >2D tensors not yet implemented",
        ));
    }

    Ok(())
}

// Helper implementations for max/min operations
fn max_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let max_val = data
        .par_iter()
        .cloned()
        .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let max_val = data
        .par_iter()
        .cloned()
        .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b));

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let max_val = data.par_iter().copied().max().unwrap_or(i32::MIN);

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let max_val = data.par_iter().copied().max().unwrap_or(i64::MIN);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

fn max_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let max_val = data.par_iter().any(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = max_val;
    Ok(())
}

// Similar implementations for min functions
fn min_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let min_val = data
        .par_iter()
        .cloned()
        .reduce(|| f32::INFINITY, |a, b| a.min(b));

    let result_slice = result_data
        .as_f32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f32 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let min_val = data
        .par_iter()
        .cloned()
        .reduce(|| f64::INFINITY, |a, b| a.min(b));

    let result_slice = result_data
        .as_f64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable f64 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let min_val = data.par_iter().copied().min().unwrap_or(i32::MAX);

    let result_slice = result_data
        .as_i32_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i32 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let min_val = data.par_iter().copied().min().unwrap_or(i64::MAX);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

fn min_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let min_val = data.par_iter().all(|&x| x);

    let result_slice = result_data
        .as_bool_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable bool slice"))?;

    result_slice[0] = min_val;
    Ok(())
}

// Placeholder implementations for argmax/argmin
fn argmax_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::NEG_INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::NEG_INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i32::MIN),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let (argmax_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i64::MIN),
        |(i1, v1), (i2, v2)| {
            if v1 >= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

fn argmax_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let argmax_idx = data.iter().position(|&x| x).unwrap_or(0);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmax_idx as i64;
    Ok(())
}

// Similar implementations for argmin
fn argmin_all_f32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f32::INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_f64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_f64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, f64::INFINITY),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_i32(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i32_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i32::MAX),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_i64(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_i64_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;

    let (argmin_idx, _) = data.par_iter().enumerate().map(|(i, &v)| (i, v)).reduce(
        || (0, i64::MAX),
        |(i1, v1), (i2, v2)| {
            if v1 <= v2 {
                (i1, v1)
            } else {
                (i2, v2)
            }
        },
    );

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

fn argmin_all_bool(tensor: &Tensor, result_data: &mut TensorData) -> Result<()> {
    let data = tensor
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;

    let argmin_idx = data.par_iter().position_first(|&x| !x).unwrap_or(0);

    let result_slice = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    result_slice[0] = argmin_idx as i64;
    Ok(())
}

// Placeholder implementations for dimensional operations
fn max_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f64::NEG_INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i32::MIN;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i64::MIN;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val = max_val.max(input[idx]);
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = false;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        max_val |= input[idx];
                        if max_val {
                            break;
                        }
                    }
                    output[o * inner + r] = max_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn min_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), tensor.dtype(), tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let output = result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f32::INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let output = result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f64::INFINITY;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            let output = result_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i32::MAX;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            let output = result_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i64::MAX;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val = min_val.min(input[idx]);
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            let output = result_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice")
            })?;

            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = true;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        min_val &= input[idx];
                        if !min_val {
                            break;
                        }
                    }
                    output[o * inner + r] = min_val;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn argmax_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Int64, tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    let output = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i32::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_val = i64::MIN;
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = d;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut max_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        if input[idx] {
                            max_idx = d;
                            break;
                        }
                    }
                    output[o * inner + r] = max_idx as i64;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Int64,
        tensor.device(),
        false,
    ))
}

fn argmin_along_dim(tensor: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    if dim >= tensor.ndim() {
        return Err(MinitensorError::index_error(dim as isize, 0, tensor.ndim()));
    }

    let input_shape = tensor.shape().dims();
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }
    let output_shape_obj = Shape::new(output_shape);
    let mut result_data =
        TensorData::zeros_on_device(output_shape_obj.numel(), DataType::Int64, tensor.device());

    let dim_size = input_shape[dim];
    let outer = input_shape[..dim].iter().product::<usize>();
    let inner = input_shape[dim + 1..].iter().product::<usize>();
    let outer_stride = dim_size * inner;

    let output = result_data
        .as_i64_slice_mut()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get mutable i64 slice"))?;

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f32::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Float64 => {
            let input = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = f64::INFINITY;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Int32 => {
            let input = tensor
                .data()
                .as_i32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i32::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Int64 => {
            let input = tensor
                .data()
                .as_i64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_val = i64::MAX;
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        let val = input[idx];
                        if val < min_val {
                            min_val = val;
                            min_idx = d;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
        DataType::Bool => {
            let input = tensor
                .data()
                .as_bool_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
            for o in 0..outer {
                for r in 0..inner {
                    let mut min_idx = 0usize;
                    for d in 0..dim_size {
                        let idx = o * outer_stride + d * inner + r;
                        if !input[idx] {
                            min_idx = d;
                            break;
                        }
                    }
                    output[o * inner + r] = min_idx as i64;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(result_data),
        output_shape_obj,
        DataType::Int64,
        tensor.device(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
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
        assert_eq!(all_res.data().as_bool_slice().unwrap()[0], false);
        assert_eq!(any_res.data().as_bool_slice().unwrap()[0], true);
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
    fn test_sum_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = sum(&t, Some(vec![0]), false).unwrap();
        assert_eq!(res.shape().dims(), &[2]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_sum_bool_error() {
        let t = create_tensor_bool(vec![true, false, true, true], vec![2, 2]);
        assert!(sum(&t, Some(vec![0]), false).is_err());
    }

    #[test]
    fn test_sum_multi_dim_error() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(sum(&t, Some(vec![0, 1]), false).is_err());
    }

    #[test]
    fn test_mean_along_dim() {
        let t = create_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let res = mean(&t, Some(vec![1]), true).unwrap();
        assert_eq!(res.shape().dims(), &[2, 1]);
        assert_eq!(res.data().as_f32_slice().unwrap(), &[1.5, 3.5]);
    }

    #[test]
    fn test_mean_int_error() {
        let t = create_tensor_i32(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(mean(&t, Some(vec![0]), false).is_err());
    }
}
