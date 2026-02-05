// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

/// Transpose operation with gradient support
pub fn transpose(tensor: &Tensor, dim0: isize, dim1: isize) -> Result<Tensor> {
    let ndim = tensor.ndim() as isize;
    let dim0 = if dim0 < 0 { dim0 + ndim } else { dim0 };
    let dim1 = if dim1 < 0 { dim1 + ndim } else { dim1 };

    if dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim {
        return Err(MinitensorError::index_error(
            dim0.max(dim1),
            0,
            ndim as usize,
        ));
    }

    if dim0 == dim1 {
        // No-op transpose
        return Ok(tensor.clone());
    }

    let dim0_usize = dim0 as usize;
    let dim1_usize = dim1 as usize;

    // Create new shape with swapped dimensions
    let mut new_shape = tensor.shape().dims().to_vec();
    new_shape.swap(dim0_usize, dim1_usize);
    let new_shape_obj = Shape::new(new_shape);

    // Create new strides with swapped dimensions
    let old_strides = tensor.strides().as_slice();
    let mut new_strides = old_strides.to_vec();
    new_strides.swap(dim0_usize, dim1_usize);

    // Create output tensor data by copying and rearranging
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    // Perform transpose based on data type
    match tensor.dtype() {
        DataType::Float32 => transpose_f32(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Float64 => transpose_f64(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Int32 => transpose_i32(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Int64 => transpose_i64(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
        DataType::Bool => transpose_bool(
            tensor,
            &mut output_data,
            &new_shape_obj,
            dim0_usize,
            dim1_usize,
        )?,
    }

    // Create output tensor
    let output = Tensor::new(
        Arc::new(output_data),
        new_shape_obj,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    // Set up gradient function if needed
    if output.requires_grad() {
        let grad_fn = Arc::new(TransposeBackward {
            dims: vec![dim0_usize, dim1_usize],
            input_id: tensor.id(),
        });

        let mut output_with_grad = output;
        output_with_grad.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output_with_grad, Some(grad_fn))?;

        Ok(output_with_grad)
    } else {
        Ok(output)
    }
}

/// Extract a diagonal from the tensor, reducing two dimensions into one.
pub fn diagonal(tensor: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
    if tensor.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "diagonal requires tensors with at least 2 dimensions",
        ));
    }

    if !tensor.device().is_cpu() {
        return Err(MinitensorError::invalid_operation(
            "diagonal currently supports only CPU tensors",
        ));
    }

    let ndim = tensor.ndim();
    let dim1 = normalize_dim(dim1, ndim)?;
    let dim2 = normalize_dim(dim2, ndim)?;
    if dim1 == dim2 {
        return Err(MinitensorError::invalid_operation(
            "diagonal dimensions must be distinct",
        ));
    }

    let dims = tensor.shape().dims();
    let strides = tensor.strides().as_slice();
    let spec = compute_diagonal_spec(dims, strides, dim1, dim2, offset)?;
    let out_shape = Shape::new(spec.output_dims.clone());
    let dtype = tensor.dtype();
    let device = tensor.device();
    let mut output_data = TensorData::zeros_on_device(out_shape.numel(), dtype, device);

    if out_shape.numel() > 0 {
        match dtype {
            DataType::Float32 => {
                let input = tensor.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor")
                })?;
                let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Float64 => {
                let input = tensor.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor")
                })?;
                let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Int32 => {
                let input = tensor.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor")
                })?;
                let output = output_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Int64 => {
                let input = tensor.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor")
                })?;
                let output = output_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
            DataType::Bool => {
                let input = tensor.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor")
                })?;
                let output = output_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice for diagonal output",
                    )
                })?;
                diagonal_copy(input, output, dims, strides, &spec);
            }
        }
    }

    let mut output = Tensor::new(
        Arc::new(output_data),
        out_shape,
        dtype,
        device,
        tensor.requires_grad(),
    );

    if tensor.requires_grad() {
        let grad_fn = Arc::new(crate::autograd::DiagonalBackward {
            input_shape: dims.to_vec(),
            input_strides: strides.to_vec(),
            input_dtype: dtype,
            dim1,
            dim2,
            offset,
            input_requires_grad: tensor.requires_grad(),
            input_id: tensor.id(),
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Sum of the diagonal elements along two dimensions.
pub fn trace(tensor: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
    let diag = diagonal(tensor, offset, dim1, dim2)?;
    if diag.ndim() == 0 {
        return Ok(diag);
    }

    reduction::sum(&diag, Some(vec![-1]), false)
}

/// Return the upper triangular part of a matrix (or batch of matrices).
pub fn triu(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    triangular_op(tensor, diagonal, true)
}

/// Return the lower triangular part of a matrix (or batch of matrices).
pub fn tril(tensor: &Tensor, diagonal: i64) -> Result<Tensor> {
    triangular_op(tensor, diagonal, false)
}

fn triangular_op(tensor: &Tensor, diagonal: i64, upper: bool) -> Result<Tensor> {
    if tensor.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "triangular operations require tensors with at least 2 dimensions",
        ));
    }

    let clamped_diagonal = diagonal.clamp(isize::MIN as i64, isize::MAX as i64) as isize;

    let mut output_data =
        TensorData::uninitialized_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    apply_triangular_mask(tensor, &mut output_data, clamped_diagonal, upper)?;

    let mut output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if tensor.requires_grad() {
        let grad_fn = Arc::new(crate::autograd::TriangularBackward {
            input_shape: tensor.shape().dims().to_vec(),
            diagonal: clamped_diagonal,
            upper,
            input_requires_grad: tensor.requires_grad(),
            input_id: tensor.id(),
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

pub(crate) fn apply_triangular_mask(
    tensor: &Tensor,
    output_data: &mut TensorData,
    diagonal: isize,
    upper: bool,
) -> Result<()> {
    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Int32 => {
            let input = tensor.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from tensor")
            })?;
            let output = output_data.as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i32 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Int64 => {
            let input = tensor.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from tensor")
            })?;
            let output = output_data.as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable i64 slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
        DataType::Bool => {
            let input = tensor.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from tensor")
            })?;
            let output = output_data.as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable bool slice for triangular output",
                )
            })?;
            copy_and_mask(input, output, tensor.shape(), diagonal, upper);
        }
    }

    Ok(())
}

fn copy_and_mask<T: Copy + Default>(
    input: &[T],
    output: &mut [T],
    shape: &Shape,
    diagonal: isize,
    upper: bool,
) {
    if input.is_empty() {
        return;
    }

    output.copy_from_slice(input);

    let dims = shape.dims();
    debug_assert!(dims.len() >= 2);
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];

    if rows == 0 || cols == 0 {
        return;
    }

    let batch = shape.numel() / (rows * cols);
    let zero = T::default();

    for b in 0..batch {
        let base = b * rows * cols;
        for r in 0..rows {
            let row_offset = base + r * cols;
            let row_idx = r as isize;
            for c in 0..cols {
                let col_idx = c as isize;
                let keep = if upper {
                    col_idx - row_idx >= diagonal
                } else {
                    col_idx - row_idx <= diagonal
                };
                if !keep {
                    output[row_offset + c] = zero;
                }
            }
        }
    }
}

// Helper functions for matrix multiplication

fn matmul_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    _output_shape: &Shape,
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

    optimized_matmul_f32(lhs_data, rhs_data, output_slice, lhs.shape(), rhs.shape())
}

fn matmul_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    _output_shape: &Shape,
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

    optimized_matmul_f64(lhs_data, rhs_data, output_slice, lhs.shape(), rhs.shape())
}

fn matmul_i32(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
}

fn matmul_i64(
    lhs: &Tensor,
    rhs: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
) -> Result<()> {
    let lhs_data = lhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from lhs tensor")
    })?;
    let rhs_data = rhs.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    naive_matmul(
        lhs_data,
        rhs_data,
        output_slice,
        lhs.shape(),
        rhs.shape(),
        output_shape,
    )
}

/// Naive matrix multiplication implementation (O(n^3)) with batch support
fn naive_matmul<T>(
    lhs_data: &[T],
    rhs_data: &[T],
    output_data: &mut [T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    _output_shape: &Shape,
) -> Result<()>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default + Send + Sync,
{
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();

    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];
    let batch = lhs_data.len() / (m * k);
    if batch == 1 && m * n * k < PAR_THRESHOLD {
        // For small single-batch matrices, avoid parallel overhead
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    let lhs_idx = i * k + l;
                    let rhs_idx = l * n + j;
                    sum = sum + lhs_data[lhs_idx] * rhs_data[rhs_idx];
                }
                output_data[i * n + j] = sum;
            }
        }
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let lhs_batch = &lhs_data[b * m * k..(b + 1) * m * k];
                let rhs_batch = &rhs_data[b * k * n..(b + 1) * k * n];
                chunk.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                    for j in 0..n {
                        let mut sum = T::default();
                        for l in 0..k {
                            let lhs_idx = i * k + l;
                            let rhs_idx = l * n + j;
                            sum = sum + lhs_batch[lhs_idx] * rhs_batch[rhs_idx];
                        }
                        row[j] = sum;
                    }
                });
            });
    }

    Ok(())
}

fn optimized_matmul_f32(
    lhs_data: &[f32],
    rhs_data: &[f32],
    output_data: &mut [f32],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<()> {
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    if m == 0 || k == 0 || n == 0 {
        // Nothing to compute for zero-sized dimensions
        return Ok(());
    }

    let batch = lhs_data.len() / (m * k);
    if batch == 1 {
        // Avoid parallel overhead for single matrix multiplication
        unsafe {
            gemm_f32(
                m,
                k,
                n,
                lhs_data.as_ptr(),
                rhs_data.as_ptr(),
                output_data.as_mut_ptr(),
            )
        };
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let a = &lhs_data[b * m * k..(b + 1) * m * k];
                let r = &rhs_data[b * k * n..(b + 1) * k * n];
                unsafe {
                    gemm_f32(m, k, n, a.as_ptr(), r.as_ptr(), chunk.as_mut_ptr());
                }
            });
    }

    Ok(())
}

fn optimized_matmul_f64(
    lhs_data: &[f64],
    rhs_data: &[f64],
    output_data: &mut [f64],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<()> {
    let lhs_dims = lhs_shape.dims();
    let rhs_dims = rhs_shape.dims();
    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

    let batch = lhs_data.len() / (m * k);
    if batch == 1 {
        unsafe {
            gemm_f64(
                m,
                k,
                n,
                lhs_data.as_ptr(),
                rhs_data.as_ptr(),
                output_data.as_mut_ptr(),
            )
        };
    } else {
        output_data
            .par_chunks_mut(m * n)
            .enumerate()
            .for_each(|(b, chunk)| {
                let a = &lhs_data[b * m * k..(b + 1) * m * k];
                let r = &rhs_data[b * k * n..(b + 1) * k * n];
                unsafe {
                    gemm_f64(m, k, n, a.as_ptr(), r.as_ptr(), chunk.as_mut_ptr());
                }
            });
    }

    Ok(())
}

// Helper functions for transpose operations

fn transpose_f32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f32 slice from input tensor")
    })?;

    let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f32 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_f64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get f64 slice from input tensor")
    })?;

    let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable f64 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_i32(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_i32_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i32 slice from input tensor")
    })?;

    let output_slice = output_data.as_i32_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i32 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_i64(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get i64 slice from input tensor")
    })?;

    let output_slice = output_data.as_i64_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable i64 slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}

fn transpose_bool(
    tensor: &Tensor,
    output_data: &mut TensorData,
    output_shape: &Shape,
    dim0: usize,
    dim1: usize,
) -> Result<()> {
    let input_data = tensor.data().as_bool_slice().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get bool slice from input tensor")
    })?;

    let output_slice = output_data.as_bool_slice_mut().ok_or_else(|| {
        MinitensorError::internal_error("Failed to get mutable bool slice from output data")
    })?;

    transpose_generic(
        input_data,
        output_slice,
        tensor.shape(),
        output_shape,
        dim0,
        dim1,
    )
}
