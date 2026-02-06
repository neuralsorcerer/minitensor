// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use num_traits::Float;

fn logaddexp_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    crate::operations::arithmetic::broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )
}

fn logaddexp_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    crate::operations::arithmetic::broadcast_binary_op(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
        |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                let max = a.max(b);
                if max.is_infinite() {
                    max
                } else {
                    let exp_a = (a - max).exp();
                    let exp_b = (b - max).exp();
                    max + (exp_a + exp_b).ln()
                }
            }
        },
    )
}

fn tanh_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f32::tanh);
    Ok(())
}

fn tanh_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, f64::tanh);
    Ok(())
}

fn sigmoid_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f32| {
        1.0 / (1.0 + (-val).exp())
    });
    Ok(())
}

fn sigmoid_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    unary_apply(input_data, output_slice, |val: f64| {
        1.0 / (1.0 + (-val).exp())
    });
    Ok(())
}

fn relu_f32(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v.is_nan() {
                    *o = v;
                } else if v > 0.0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0.0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v.is_nan() {
                *o = v;
            } else if v > 0.0 {
                *o = v;
                *m = true;
            } else {
                *o = 0.0;
            }
        }
    }
    Ok(mask)
}

fn relu_f64(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v.is_nan() {
                    *o = v;
                } else if v > 0.0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0.0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v.is_nan() {
                *o = v;
            } else if v > 0.0 {
                *o = v;
                *m = true;
            } else {
                *o = 0.0;
            }
        }
    }
    Ok(mask)
}

fn relu_i32(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v > 0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v > 0 {
                *o = v;
                *m = true;
            } else {
                *o = 0;
            }
        }
    }
    Ok(mask)
}

fn relu_i64(tensor: &Tensor, output_data: &mut TensorData) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;
    let len = input_data.len();
    let mut mask = vec![false; len];
    if len >= PAR_THRESHOLD {
        output_slice
            .par_iter_mut()
            .zip(input_data.par_iter())
            .zip(mask.par_iter_mut())
            .for_each(|((o, &v), m)| {
                if v > 0 {
                    *o = v;
                    *m = true;
                } else {
                    *o = 0;
                }
            });
    } else {
        for ((o, &v), m) in output_slice
            .iter_mut()
            .zip(input_data.iter())
            .zip(mask.iter_mut())
        {
            if v > 0 {
                *o = v;
                *m = true;
            } else {
                *o = 0;
            }
        }
    }
    Ok(mask)
}

fn hardshrink_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    lambd: f32,
    store_mask: bool,
) -> Result<Option<Vec<bool>>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let mut mask = if store_mask {
        Some(Vec::with_capacity(input_data.len()))
    } else {
        None
    };

    for (&value, out_slot) in input_data.iter().zip(output_slice.iter_mut()) {
        let keep = value > lambd || value < -lambd;
        *out_slot = if keep { value } else { 0.0 };
        if let Some(ref mut mask_vec) = mask {
            mask_vec.push(keep);
        }
    }

    Ok(mask)
}

fn hardshrink_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    lambd: f64,
    store_mask: bool,
) -> Result<Option<Vec<bool>>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let mut mask = if store_mask {
        Some(Vec::with_capacity(input_data.len()))
    } else {
        None
    };

    for (&value, out_slot) in input_data.iter().zip(output_slice.iter_mut()) {
        let keep = value > lambd || value < -lambd;
        *out_slot = if keep { value } else { 0.0 };
        if let Some(ref mut mask_vec) = mask {
            mask_vec.push(keep);
        }
    }

    Ok(mask)
}

fn leaky_relu_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f32,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let len = input_data.len();
    let mut mask = vec![false; len];
    let mask_ptr = mask.as_mut_ptr() as usize;
    let in_ptr = input_data.as_ptr() as usize;
    let out_ptr = output_slice.as_mut_ptr() as usize;
    (0..len).into_par_iter().for_each(|i| unsafe {
        let in_ptr = in_ptr as *const f32;
        let out_ptr = out_ptr as *mut f32;
        let mask_ptr = mask_ptr as *mut bool;
        let val = *in_ptr.add(i);
        if val >= 0.0 {
            *out_ptr.add(i) = val;
            *mask_ptr.add(i) = true;
        } else {
            *out_ptr.add(i) = negative_slope * val;
        }
    });
    Ok(mask)
}

fn leaky_relu_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    negative_slope: f64,
) -> Result<Vec<bool>> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let len = input_data.len();
    let mut mask = vec![false; len];
    let mask_ptr = mask.as_mut_ptr() as usize;
    let in_ptr = input_data.as_ptr() as usize;
    let out_ptr = output_slice.as_mut_ptr() as usize;
    (0..len).into_par_iter().for_each(|i| unsafe {
        let in_ptr = in_ptr as *const f64;
        let out_ptr = out_ptr as *mut f64;
        let mask_ptr = mask_ptr as *mut bool;
        let val = *in_ptr.add(i);
        if val >= 0.0 {
            *out_ptr.add(i) = val;
            *mask_ptr.add(i) = true;
        } else {
            *out_ptr.add(i) = negative_slope * val;
        }
    });
    Ok(mask)
}

fn softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    if dim_size == 0 {
        return Ok(());
    }

    // Compute the number of groups before and after the softmax dimension. This
    // allows us to iterate over all slices along `dim` for tensors of arbitrary
    // rank using row-major indexing.
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let val = (in_block[idx] - max_val).exp();
                    out_block[idx] = val;
                    sum += val;
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] /= sum;
                }
            }
        });

    Ok(())
}

fn softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let dim_size = dims[dim];

    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    max_val = max_val.max(in_block[idx]);
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let val = (in_block[idx] - max_val).exp();
                    out_block[idx] = val;
                    sum += val;
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] /= sum;
                }
            }
        });

    Ok(())
}

fn broadcast_mask_index(
    linear_idx: usize,
    output_dims: &[usize],
    output_strides: &[usize],
    mask_dims: &[usize],
    mask_strides: &[usize],
) -> usize {
    if mask_dims.is_empty() {
        return 0;
    }

    let output_ndim = output_dims.len();
    let mask_ndim = mask_dims.len();
    let mut mask_index = 0usize;

    for i in 0..mask_ndim {
        let output_dim_idx = output_ndim - 1 - i;
        let mask_dim_idx = mask_ndim - 1 - i;
        let stride = output_strides[output_dim_idx];
        let coord = if stride == 0 {
            0
        } else {
            (linear_idx / stride) % output_dims[output_dim_idx]
        };
        let mask_dim = mask_dims[mask_dim_idx];
        let mask_coord = if mask_dim == 1 { 0 } else { coord };
        mask_index += mask_coord * mask_strides[mask_dim_idx];
    }

    mask_index
}

fn masked_softmax_f32(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;
    let mask_data = mask.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from mask tensor")
    })?;
    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let mask_dims = mask.shape().dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;
    let output_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(tensor.shape()))
    };
    let mask_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(mask.shape()))
    };

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = f32::NEG_INFINITY;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if !masked {
                        has_unmasked = true;
                        max_val = max_val.max(in_block[idx]);
                    }
                }
                if !has_unmasked {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let val = (in_block[idx] - max_val).exp();
                        out_block[idx] = val;
                        sum += val;
                    }
                }
                if sum != 0.0 {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] /= sum;
                    }
                }
            }
        });

    Ok(())
}

fn masked_softmax_f64(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;
    let mask_data = mask.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from mask tensor")
    })?;
    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    let dims = tensor.shape().dims();
    let mask_dims = mask.shape().dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;
    let output_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(tensor.shape()))
    };
    let mask_strides = if same_shape {
        None
    } else {
        Some(Strides::from_shape(mask.shape()))
    };

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = f64::NEG_INFINITY;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if !masked {
                        has_unmasked = true;
                        max_val = max_val.max(in_block[idx]);
                    }
                }
                if !has_unmasked {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask_data[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.as_ref().unwrap().as_slice(),
                            mask_dims,
                            mask_strides.as_ref().unwrap().as_slice(),
                        );
                        mask_data[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let val = (in_block[idx] - max_val).exp();
                        out_block[idx] = val;
                        sum += val;
                    }
                }
                if sum != 0.0 {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] /= sum;
                    }
                }
            }
        });

    Ok(())
}

fn log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    dims: &[usize],
    dim: usize,
    neg_inf: T,
) -> Result<()> {
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .for_each(|(in_block, out_block)| {
            for a in 0..after {
                let base = a;
                let mut max_val = neg_inf;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let val = in_block[idx];
                    if val > max_val {
                        max_val = val;
                    }
                }
                if max_val.is_infinite() && max_val.is_sign_negative() {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = neg_inf;
                    }
                    continue;
                }
                let mut sum = T::zero();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum = sum + (in_block[idx] - max_val).exp();
                }
                let logsum = sum.ln() + max_val;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = in_block[idx] - logsum;
                }
            }
        });

    Ok(())
}

fn masked_log_softmax_core<T: Float + Send + Sync>(
    input_data: &[T],
    output_slice: &mut [T],
    mask_data: &[bool],
    tensor_shape: &Shape,
    mask_shape: &Shape,
    dim: usize,
    neg_inf: T,
) -> Result<()> {
    let dims = tensor_shape.dims();
    let mask_dims = mask_shape.dims();
    let dim_size = dims[dim];
    if dim_size == 0 {
        return Ok(());
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = mask_dims == dims;

    if same_shape {
        input_data
            .par_chunks(group)
            .zip(output_slice.par_chunks_mut(group))
            .enumerate()
            .for_each(|(block_idx, (in_block, out_block))| {
                let block_offset = block_idx * group;
                for a in 0..after {
                    let base = a;
                    let mut max_val = neg_inf;
                    let mut has_unmasked = false;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if !mask_data[linear_idx] {
                            has_unmasked = true;
                            let val = in_block[idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    if !has_unmasked || (max_val.is_infinite() && max_val.is_sign_negative()) {
                        for k in 0..dim_size {
                            let idx = base + k * after;
                            out_block[idx] = neg_inf;
                        }
                        continue;
                    }
                    let mut sum = T::zero();
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if !mask_data[linear_idx] {
                            sum = sum + (in_block[idx] - max_val).exp();
                        }
                    }
                    let logsum = sum.ln() + max_val;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        if mask_data[linear_idx] {
                            out_block[idx] = neg_inf;
                        } else {
                            out_block[idx] = in_block[idx] - logsum;
                        }
                    }
                }
            });
        return Ok(());
    }

    let output_strides = Strides::from_shape(tensor_shape);
    let mask_strides = Strides::from_shape(mask_shape);
    let output_stride_slice = output_strides.as_slice();
    let mask_stride_slice = mask_strides.as_slice();

    input_data
        .par_chunks(group)
        .zip(output_slice.par_chunks_mut(group))
        .enumerate()
        .for_each(|(block_idx, (in_block, out_block))| {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut max_val = neg_inf;
                let mut has_unmasked = false;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if !mask_data[mask_index] {
                        has_unmasked = true;
                        let val = in_block[idx];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                if !has_unmasked || (max_val.is_infinite() && max_val.is_sign_negative()) {
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = neg_inf;
                    }
                    continue;
                }
                let mut sum = T::zero();
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if !mask_data[mask_index] {
                        sum = sum + (in_block[idx] - max_val).exp();
                    }
                }
                let logsum = sum.ln() + max_val;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let mask_index = broadcast_mask_index(
                        linear_idx,
                        dims,
                        output_stride_slice,
                        mask_dims,
                        mask_stride_slice,
                    );
                    if mask_data[mask_index] {
                        out_block[idx] = neg_inf;
                    } else {
                        out_block[idx] = in_block[idx] - logsum;
                    }
                }
            }
        });

    Ok(())
}

macro_rules! masked_log_softmax_impl {
    (
        $tensor:expr,
        $mask:expr,
        $output_data:expr,
        $dim:expr,
        $input_ty:ty,
        $as_input:ident,
        $as_output:ident,
        $neg_inf:expr
    ) => {{
        let input_data = $tensor.data().$as_input().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get input slice from tensor")
        })?;
        let mask_data = $mask.data().as_bool_slice().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get bool slice from mask tensor")
        })?;
        let output_slice = $output_data.$as_output().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable output slice from data")
        })?;

        masked_log_softmax_core(
            input_data,
            output_slice,
            mask_data,
            $tensor.shape(),
            $mask.shape(),
            $dim,
            $neg_inf,
        )
    }};
}

macro_rules! log_softmax_impl {
    (
        $tensor:expr,
        $output_data:expr,
        $dim:expr,
        $input_ty:ty,
        $as_input:ident,
        $as_output:ident,
        $neg_inf:expr
    ) => {{
        let input_data = $tensor.data().$as_input().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get input slice from tensor")
        })?;
        let output_slice = $output_data.$as_output().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable output slice from data")
        })?;

        let dims = $tensor.shape().dims();
        log_softmax_core(input_data, output_slice, dims, $dim, $neg_inf)
    }};
}

fn masked_log_softmax_f32(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    masked_log_softmax_impl!(
        tensor,
        mask,
        output_data,
        dim,
        f32,
        as_f32_slice,
        as_f32_slice_mut,
        f32::NEG_INFINITY
    )
}

fn masked_log_softmax_f64(
    tensor: &Tensor,
    mask: &Tensor,
    output_data: &mut TensorData,
    dim: usize,
) -> Result<()> {
    masked_log_softmax_impl!(
        tensor,
        mask,
        output_data,
        dim,
        f64,
        as_f64_slice,
        as_f64_slice_mut,
        f64::NEG_INFINITY
    )
}

fn log_softmax_f32(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    log_softmax_impl!(
        tensor,
        output_data,
        dim,
        f32,
        as_f32_slice,
        as_f32_slice_mut,
        f32::NEG_INFINITY
    )
}

fn log_softmax_f64(tensor: &Tensor, output_data: &mut TensorData, dim: usize) -> Result<()> {
    log_softmax_impl!(
        tensor,
        output_data,
        dim,
        f64,
        as_f64_slice,
        as_f64_slice_mut,
        f64::NEG_INFINITY
    )
}
