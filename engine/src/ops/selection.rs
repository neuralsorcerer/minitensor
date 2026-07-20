// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{WhereBackward, add_to_graph},
    device::Device,
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_binary_operands},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

/// Select elements from ``input`` or ``other`` based on ``condition``.
///
/// ``condition`` must be a boolean tensor. All tensors must reside on the same
/// device and have broadcastable shapes. The result has the broadcasted shape of
/// the three operands and takes values from ``input`` where ``condition`` is
/// true and from ``other`` where it is false.
pub fn where_op(condition: &Tensor, input: &Tensor, other: &Tensor) -> Result<Tensor> {
    if condition.device() != input.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", condition.device()),
            format!("{:?}", input.device()),
        ));
    }
    if condition.device() != other.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", condition.device()),
            format!("{:?}", other.device()),
        ));
    }
    if input.device() != other.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", other.device()),
        ));
    }

    if condition.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "where requires condition tensor of dtype bool",
        ));
    }

    let (input_cast, other_cast, result_dtype) =
        coerce_binary_operands(input, other, BinaryOpKind::Add)?;
    let input_tensor = input_cast.as_ref();
    let other_tensor = other_cast.as_ref();

    let tmp_shape = condition.shape().broadcast_with(input_tensor.shape())?;
    let output_shape = tmp_shape.broadcast_with(other_tensor.shape())?;

    /// One dtype arm: run the where selection into a fresh buffer.
    macro_rules! where_arm {
        ($accessor:ident, $ty:ty, $tyname:literal) => {{
            let input_slice = input_tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input tensor"
                ))
            })?;
            let other_slice = other_tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from other tensor"
                ))
            })?;
            let cond_slice = condition.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from condition tensor")
            })?;
            TensorData::from_vec::<$ty>(
                where_map(
                    cond_slice,
                    input_slice,
                    other_slice,
                    condition.shape(),
                    input_tensor.shape(),
                    other_tensor.shape(),
                    &output_shape,
                )?,
                result_dtype,
                input_tensor.device(),
            )
        }};
    }

    let output_data = match result_dtype {
        DataType::Float32 => where_arm!(as_f32_slice, f32, "f32"),
        DataType::Float64 => where_arm!(as_f64_slice, f64, "f64"),
        DataType::Int32 => where_arm!(as_i32_slice, i32, "i32"),
        DataType::Int64 => where_arm!(as_i64_slice, i64, "i64"),
        DataType::Bool => where_arm!(as_bool_slice, bool, "bool"),
    };

    let requires_grad = input.requires_grad() || other.requires_grad();
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        input_tensor.device(),
        requires_grad,
    );

    if requires_grad {
        let grad_fn = Arc::new(WhereBackward {
            condition: condition.detach(),
            input_shape: input.shape().dims().to_vec(),
            other_shape: other.shape().dims().to_vec(),
            input_requires_grad: input.requires_grad(),
            other_requires_grad: other.requires_grad(),
            input_ids: [input.id(), other.id()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Fill elements of `input` where `mask` is `True` with values from `value`.
pub fn masked_fill(input: &Tensor, mask: &Tensor, value: &Tensor) -> Result<Tensor> {
    if mask.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "masked_fill mask must have bool dtype",
        ));
    }

    if input.device() != mask.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", mask.device()),
        ));
    }

    if input.device() != value.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", value.device()),
        ));
    }

    if input.dtype() != value.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", input.dtype()),
            format!("{:?}", value.dtype()),
        ));
    }

    where_op(mask, value, input)
}

/// Scalar convenience for [`masked_fill`].
pub fn masked_fill_scalar(input: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor> {
    let scalar = scalar_tensor(value, input.dtype(), input.device())?;
    masked_fill(input, mask, &scalar)
}

/// NumPy-style boolean indexing: `mask`'s shape must equal `input`'s leading
/// `mask.ndim()` dimensions, and the result stacks the selected trailing
/// blocks — shape `[n_true] + input.shape[mask.ndim():]`. A full-shape mask
/// therefore selects individual elements into a 1-D tensor (PyTorch's
/// `masked_select`), while a 1-D mask over a matrix selects rows.
pub fn masked_index(input: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if mask.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "boolean index mask must have bool dtype",
        ));
    }
    if input.device() != mask.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", mask.device()),
        ));
    }

    let in_dims = input.shape().dims();
    let m_dims = mask.shape().dims();
    if m_dims.len() > in_dims.len() || in_dims[..m_dims.len()] != *m_dims {
        return Err(MinitensorError::invalid_argument(format!(
            "boolean index mask shape {:?} must match the leading dimensions of tensor shape {:?}",
            m_dims, in_dims
        )));
    }

    let inner: usize = in_dims[m_dims.len()..].iter().product();
    let input_c = input.contiguous()?;
    let mask_c = mask.contiguous()?;
    let mask_slice = mask_c
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from mask"))?;

    let selected: Vec<usize> = mask_slice
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| m.then_some(i))
        .collect();

    let mut out_dims = vec![selected.len()];
    out_dims.extend_from_slice(&in_dims[m_dims.len()..]);
    let out_shape = Shape::new(out_dims);

    let mut output_data =
        TensorData::zeros_on_device(out_shape.numel(), input.dtype(), input.device());

    /// Copies the selected trailing blocks for one dtype.
    macro_rules! gather_arm {
        ($accessor:ident, $accessor_mut:ident, $tyname:literal) => {{
            let src = input_c.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from input"
                ))
            })?;
            let dst = output_data.$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice for output"
                ))
            })?;
            for (k, &blk) in selected.iter().enumerate() {
                dst[k * inner..(k + 1) * inner]
                    .copy_from_slice(&src[blk * inner..(blk + 1) * inner]);
            }
        }};
    }

    if inner > 0 {
        match input.dtype() {
            DataType::Float32 => gather_arm!(as_f32_slice, as_f32_slice_mut, "f32"),
            DataType::Float64 => gather_arm!(as_f64_slice, as_f64_slice_mut, "f64"),
            DataType::Int32 => gather_arm!(as_i32_slice, as_i32_slice_mut, "i32"),
            DataType::Int64 => gather_arm!(as_i64_slice, as_i64_slice_mut, "i64"),
            DataType::Bool => gather_arm!(as_bool_slice, as_bool_slice_mut, "bool"),
        }
    }

    let requires_grad = input.requires_grad() && input.dtype() != DataType::Bool;
    let mut output = Tensor::new(
        Arc::new(output_data),
        out_shape,
        input.dtype(),
        input.device(),
        requires_grad,
    );

    if requires_grad {
        let grad_fn = Arc::new(crate::autograd::MaskedIndexBackward {
            mask: mask_c.detach(),
            input_shape: input.shape().clone(),
            inner,
            input_id: input.id(),
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// NumPy-style boolean-mask assignment: writes `values` into the blocks of
/// `input` selected by `mask` (same leading-dimensions rule as
/// [`masked_index`]). `values` is cast to `input`'s dtype and must broadcast
/// to the selection shape `[n_true] + input.shape[mask.ndim():]` — so a
/// scalar fills every selected block, a trailing-shaped tensor is copied to
/// each block, and a `[n_true, ...]` tensor assigns block-by-block. This is
/// an in-place data mutation and does not participate in autograd, mirroring
/// `index_assign`.
pub fn masked_index_assign(input: &mut Tensor, mask: &Tensor, values: &Tensor) -> Result<()> {
    if mask.dtype() != DataType::Bool {
        return Err(MinitensorError::invalid_operation(
            "boolean index mask must have bool dtype",
        ));
    }
    if input.device() != mask.device() || input.device() != values.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", mask.device()),
        ));
    }

    let in_dims = input.shape().dims().to_vec();
    let m_dims = mask.shape().dims();
    if m_dims.len() > in_dims.len() || in_dims[..m_dims.len()] != *m_dims {
        return Err(MinitensorError::invalid_argument(format!(
            "boolean index mask shape {:?} must match the leading dimensions of tensor shape {:?}",
            m_dims, in_dims
        )));
    }

    let inner: usize = in_dims[m_dims.len()..].iter().product();
    let mask_c = mask.contiguous()?;
    let mask_slice = mask_c
        .data()
        .as_bool_slice()
        .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice from mask"))?;
    let selected: Vec<usize> = mask_slice
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| m.then_some(i))
        .collect();

    let mut sel_dims = vec![selected.len()];
    sel_dims.extend_from_slice(&in_dims[m_dims.len()..]);
    let sel_shape = Shape::new(sel_dims);

    let values_cast = if values.dtype() == input.dtype() {
        values.clone()
    } else {
        values.astype(input.dtype())?
    };
    let values_b = if values_cast.shape() == &sel_shape {
        values_cast.contiguous()?
    } else {
        let target = values_cast.shape().broadcast_with(&sel_shape)?;
        if target != sel_shape {
            return Err(MinitensorError::invalid_argument(format!(
                "cannot broadcast values of shape {:?} to boolean-mask selection shape {:?}",
                values_cast.shape().dims(),
                sel_shape.dims()
            )));
        }
        let dims: Vec<isize> = sel_shape.dims().iter().map(|&d| d as isize).collect();
        values_cast.expand(dims)?.contiguous()?
    };

    if inner == 0 || selected.is_empty() {
        return Ok(());
    }

    /// Writes the selected trailing blocks for one dtype.
    macro_rules! scatter_assign_arm {
        ($accessor:ident, $accessor_mut:ident, $tyname:literal) => {{
            // Read the source first: with self-aliasing assignments
            // (`t[mask] = t`) `data_mut`'s copy-on-write would otherwise race
            // the read view. Copying the source up front keeps it sound.
            let src: Vec<_> = values_b
                .data()
                .$accessor()
                .ok_or_else(|| {
                    MinitensorError::internal_error(concat!(
                        "Failed to get ",
                        $tyname,
                        " slice from values"
                    ))
                })?
                .to_vec();
            let dst = input.data_mut().$accessor_mut().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get mutable ",
                    $tyname,
                    " slice from input"
                ))
            })?;
            for (k, &blk) in selected.iter().enumerate() {
                dst[blk * inner..(blk + 1) * inner]
                    .copy_from_slice(&src[k * inner..(k + 1) * inner]);
            }
        }};
    }

    match input.dtype() {
        DataType::Float32 => scatter_assign_arm!(as_f32_slice, as_f32_slice_mut, "f32"),
        DataType::Float64 => scatter_assign_arm!(as_f64_slice, as_f64_slice_mut, "f64"),
        DataType::Int32 => scatter_assign_arm!(as_i32_slice, as_i32_slice_mut, "i32"),
        DataType::Int64 => scatter_assign_arm!(as_i64_slice, as_i64_slice_mut, "i64"),
        DataType::Bool => scatter_assign_arm!(as_bool_slice, as_bool_slice_mut, "bool"),
    }

    Ok(())
}

/// Select `input[i]` where `condition[i]` else `other[i]`, with NumPy
/// broadcasting, into a fresh fully-initialized buffer. Parallel above the
/// binary threshold on both the same-shape and the broadcast path (the
/// previous fill-style kernel was sequential on both).
fn where_map<T: Copy + Send + Sync>(
    condition: &[bool],
    input: &[T],
    other: &[T],
    condition_shape: &Shape,
    input_shape: &Shape,
    other_shape: &Shape,
    output_shape: &Shape,
) -> Result<Vec<T>> {
    use crate::ops::map::{BINARY_PAR_THRESHOLD, PAR_CHUNK, build_vec_with};
    use std::mem::MaybeUninit;

    let numel = output_shape.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let output_dims = output_shape.dims();
    let rank = output_dims.len();

    let same_shape = condition_shape.dims() == output_dims
        && input_shape.dims() == output_dims
        && other_shape.dims() == output_dims
        && condition.len() == numel
        && input.len() == numel
        && other.len() == numel;

    if same_shape {
        let fill_chunk = |start: usize, chunk: &mut [MaybeUninit<T>]| {
            for (k, slot) in chunk.iter_mut().enumerate() {
                let i = start + k;
                slot.write(if condition[i] { input[i] } else { other[i] });
            }
        };
        // SAFETY: the chunks partition the spare slice and every element of
        // each chunk is written.
        let out = unsafe {
            build_vec_with::<T, std::convert::Infallible, _>(numel, |spare| {
                if numel < BINARY_PAR_THRESHOLD {
                    fill_chunk(0, spare);
                } else {
                    spare
                        .par_chunks_mut(PAR_CHUNK)
                        .enumerate()
                        .for_each(|(ci, chunk)| fill_chunk(ci * PAR_CHUNK, chunk));
                }
                Ok(())
            })
            .unwrap_or_else(|e| match e {})
        };
        return Ok(out);
    }

    let cond_strides = Strides::from_shape(condition_shape);
    let input_strides = Strides::from_shape(input_shape);
    let other_strides = Strides::from_shape(other_shape);

    let cond_dims = condition_shape.dims();
    let input_dims = input_shape.dims();
    let other_dims = other_shape.dims();

    let cond_stride_slice = cond_strides.as_slice();
    let input_stride_slice = input_strides.as_slice();
    let other_stride_slice = other_strides.as_slice();

    let mut cond_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];
    let mut input_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];
    let mut other_aligned: SmallVec<[usize; 8]> = smallvec![0; rank];

    let cond_offset = rank.saturating_sub(cond_dims.len());
    for (i, &dim) in cond_dims.iter().enumerate() {
        cond_aligned[cond_offset + i] = if dim == 1 { 0 } else { cond_stride_slice[i] };
    }

    let input_offset = rank.saturating_sub(input_dims.len());
    for (i, &dim) in input_dims.iter().enumerate() {
        input_aligned[input_offset + i] = if dim == 1 { 0 } else { input_stride_slice[i] };
    }

    let other_offset = rank.saturating_sub(other_dims.len());
    for (i, &dim) in other_dims.iter().enumerate() {
        other_aligned[other_offset + i] = if dim == 1 { 0 } else { other_stride_slice[i] };
    }

    let fill_chunk = |start: usize, chunk: &mut [MaybeUninit<T>]| {
        let mut coord: SmallVec<[usize; 8]> = smallvec![0; rank];
        let mut tmp = start;
        let mut cond_index = 0usize;
        let mut input_index = 0usize;
        let mut other_index = 0usize;
        for dim in (0..rank).rev() {
            coord[dim] = tmp % output_dims[dim];
            tmp /= output_dims[dim];
            cond_index += coord[dim] * cond_aligned[dim];
            input_index += coord[dim] * input_aligned[dim];
            other_index += coord[dim] * other_aligned[dim];
        }
        for slot in chunk.iter_mut() {
            slot.write(if condition[cond_index] {
                input[input_index]
            } else {
                other[other_index]
            });
            for dim in (0..rank).rev() {
                coord[dim] += 1;
                cond_index += cond_aligned[dim];
                input_index += input_aligned[dim];
                other_index += other_aligned[dim];
                if coord[dim] < output_dims[dim] {
                    break;
                }
                coord[dim] = 0;
                cond_index -= cond_aligned[dim] * output_dims[dim];
                input_index -= input_aligned[dim] * output_dims[dim];
                other_index -= other_aligned[dim] * output_dims[dim];
            }
        }
    };

    // SAFETY: the chunks partition the spare slice and the walker writes
    // every element of each chunk.
    let out = unsafe {
        build_vec_with::<T, std::convert::Infallible, _>(numel, |spare| {
            if numel < BINARY_PAR_THRESHOLD {
                fill_chunk(0, spare);
            } else {
                spare
                    .par_chunks_mut(PAR_CHUNK)
                    .enumerate()
                    .for_each(|(ci, chunk)| fill_chunk(ci * PAR_CHUNK, chunk));
            }
            Ok(())
        })
        .unwrap_or_else(|e| match e {})
    };
    Ok(out)
}

fn scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let data = match dtype {
        DataType::Float32 => TensorData::from_vec_f32(vec![value as f32], device),
        DataType::Float64 => TensorData::from_vec_f64(vec![value], device),
        DataType::Int32 => TensorData::from_vec_i32(vec![value as i32], device),
        DataType::Int64 => TensorData::from_vec_i64(vec![value as i64], device),
        DataType::Bool => TensorData::from_vec_bool(vec![value != 0.0], device),
    };

    Ok(Tensor::new(
        Arc::new(data),
        Shape::scalar(),
        dtype,
        device,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{device::Device, tensor::TensorData};

    fn tensor_from_vec_bool(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        let data = TensorData::from_vec_bool(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Bool, Device::cpu(), false)
    }

    fn tensor_from_vec_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        let data = TensorData::from_vec_i32(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Int32, Device::cpu(), false)
    }

    #[test]
    fn test_where_basic() {
        let condition = tensor_from_vec_bool(vec![true, false, true], vec![3]);
        let input = tensor_from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let other = tensor_from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]);

        let result = where_op(&condition, &input, &other).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_where_broadcasting() {
        let condition = tensor_from_vec_bool(vec![true, false], vec![2, 1]);
        let input = tensor_from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let other = tensor_from_vec_f32(vec![10.0, 20.0], vec![1, 2]);

        let result = where_op(&condition, &input, &other).unwrap();
        let data = result.data().as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 2.0, 10.0, 20.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_where_condition_type_error() {
        let condition = tensor_from_vec_f32(vec![0.0, 1.0], vec![2]);
        let input = tensor_from_vec_f32(vec![1.0, 2.0], vec![2]);
        let other = tensor_from_vec_f32(vec![3.0, 4.0], vec![2]);
        assert!(where_op(&condition, &input, &other).is_err());
    }

    #[test]
    fn test_where_dtype_promotion() {
        let condition = tensor_from_vec_bool(vec![true, false], vec![2]);
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        let other = tensor_from_vec_f32(vec![0.5, 1.5], vec![2]);

        let result = where_op(&condition, &input, &other).unwrap();
        assert_eq!(result.dtype(), DataType::Float32);
        let values = result.data().as_f32_slice().unwrap();
        assert_eq!(values, &[1.0, 1.5]);
    }

    #[test]
    fn test_masked_fill_scalar() {
        let input = tensor_from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let mask = tensor_from_vec_bool(vec![true, false, true], vec![3]);

        let result = masked_fill_scalar(&input, &mask, 0.5).unwrap();
        let values = result.data().as_f32_slice().unwrap();
        assert_eq!(values, &[0.5, 2.0, 0.5]);
    }

    #[test]
    fn test_masked_fill_tensor_broadcast() {
        let input = tensor_from_vec_i32(vec![1, 2, 3, 4], vec![2, 2]);
        let mask = tensor_from_vec_bool(vec![true, false], vec![1, 2]);
        let fill = tensor_from_vec_i32(vec![9], vec![]);

        let result = masked_fill(&input, &mask, &fill).unwrap();
        let values = result.data().as_i32_slice().unwrap();
        assert_eq!(values, &[9, 2, 9, 4]);
    }

    #[test]
    fn test_masked_index_full_and_prefix() {
        let input = tensor_from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Full-shape mask selects individual elements into a 1-D tensor.
        let full = tensor_from_vec_bool(vec![true, false, true, false, false, true], vec![2, 3]);
        let out = masked_index(&input, &full).unwrap();
        assert_eq!(out.shape().dims(), &[3]);
        assert_eq!(out.data().as_f32_slice().unwrap(), &[1.0, 3.0, 6.0]);

        // A 1-D prefix mask selects rows.
        let rows = tensor_from_vec_bool(vec![false, true], vec![2]);
        let out = masked_index(&input, &rows).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3]);
        assert_eq!(out.data().as_f32_slice().unwrap(), &[4.0, 5.0, 6.0]);

        // All-false masks give an empty leading dim.
        let none = tensor_from_vec_bool(vec![false, false], vec![2]);
        let out = masked_index(&input, &none).unwrap();
        assert_eq!(out.shape().dims(), &[0, 3]);
    }

    #[test]
    fn test_masked_index_rejects_bad_masks() {
        let input = tensor_from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let wrong_len = tensor_from_vec_bool(vec![true, false, true], vec![3]);
        assert!(masked_index(&input, &wrong_len).is_err());
        let not_bool = tensor_from_vec_f32(vec![1.0, 0.0], vec![2]);
        assert!(masked_index(&input, &not_bool).is_err());
    }
}
