// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    autograd::{
        AddBackward, DivBackward, MulBackward, NegBackward, RemainderBackward, SubBackward,
        add_to_graph,
    },
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_binary_operands},
    ops::kernels::*,
    tensor::{DataType, Tensor, TensorData},
};
use rayon::prelude::*;
use std::sync::Arc;

pub(crate) use crate::ops::map::PAR_THRESHOLD;
use crate::ops::map::unary_map;

/// Element-wise addition with broadcasting support
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(AddBackward {
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_ids: [lhs.id(), rhs.id()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Perform element-wise addition based on data type; the kernel produces
    // the output buffer directly (no zero-init pass).
    let output_data = match result_dtype {
        DataType::Float32 => add_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => add_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 => add_i32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => add_i64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => add_bool_direct(lhs_ref, rhs_ref, &output_shape)?,
    };

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(AddBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));

        // Add to computation graph
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// In-place element-wise addition used for gradient accumulation
pub fn add_inplace(lhs: &mut Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.shape() != rhs.shape() {
        return Err(MinitensorError::shape_mismatch(
            lhs.shape().dims().to_vec(),
            rhs.shape().dims().to_vec(),
        ));
    }
    if lhs.dtype() != rhs.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", lhs.dtype()),
            format!("{:?}", rhs.dtype()),
        ));
    }
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }
    if std::sync::Arc::strong_count(lhs.data()) > 1 {
        // Fallback to out-of-place addition if data is shared
        let tmp = add(lhs, rhs)?;
        *lhs = tmp;
        return Ok(());
    }

    match lhs.dtype() {
        DataType::Float32 => {
            let lhs_slice = lhs.data_mut().as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from rhs tensor")
            })?;
            binary_assign_slices(lhs_slice, rhs_slice, |l, r| l + r);
        }
        DataType::Float64 => {
            let lhs_slice = lhs.data_mut().as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from rhs tensor")
            })?;
            binary_assign_slices(lhs_slice, rhs_slice, |l, r| l + r);
        }
        DataType::Int32 => {
            let lhs_slice = lhs.data_mut().as_i32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i32 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_i32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
            })?;
            binary_assign_slices(lhs_slice, rhs_slice, |l, r| l + r);
        }
        DataType::Int64 => {
            let lhs_slice = lhs.data_mut().as_i64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable i64 slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_i64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
            })?;
            binary_assign_slices(lhs_slice, rhs_slice, |l, r| l + r);
        }
        DataType::Bool => {
            let lhs_slice = lhs.data_mut().as_bool_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable bool slice from lhs tensor")
            })?;
            let rhs_slice = rhs.data().as_bool_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get bool slice from rhs tensor")
            })?;
            binary_assign_slices(lhs_slice, rhs_slice, |l, r| l || r);
        }
    }
    Ok(())
}

/// Apply `op` element-wise, writing the result into `lhs`.
///
/// Safe replacement for the previous raw-pointer parallel loops: chunked
/// `rayon` iteration keeps bounds information visible to the compiler (so the
/// inner loops still vectorise) without any `unsafe`.
#[inline]
fn binary_assign_slices<T: Copy + Send + Sync>(
    lhs: &mut [T],
    rhs: &[T],
    op: impl Fn(T, T) -> T + Send + Sync,
) {
    debug_assert_eq!(lhs.len(), rhs.len());
    const CHUNK: usize = 4096;
    if lhs.len() < PAR_THRESHOLD {
        for (l, &r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = op(*l, r);
        }
    } else {
        lhs.par_chunks_mut(CHUNK)
            .zip(rhs.par_chunks(CHUNK))
            .for_each(|(lhs_chunk, rhs_chunk)| {
                for (l, &r) in lhs_chunk.iter_mut().zip(rhs_chunk.iter()) {
                    *l = op(*l, r);
                }
            });
    }
}

/// Element-wise subtraction with broadcasting support
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Sub)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(SubBackward {
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_ids: [lhs.id(), rhs.id()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Perform element-wise subtraction based on data type
    let output_data = match result_dtype {
        DataType::Float32 => sub_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => sub_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 => sub_i32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => sub_i64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => unreachable!("boolean subtraction should be rejected during coercion"),
    };

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(SubBackward {
            input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Element-wise multiplication with broadcasting support
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Mul)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(MulBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                input_ids: [lhs.id(), rhs.id()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Perform element-wise multiplication based on data type
    let output_data = match result_dtype {
        DataType::Float32 => mul_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => mul_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 => mul_i32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => mul_i64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => mul_bool_direct(lhs_ref, rhs_ref, &output_shape)?,
    };

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Element-wise division with broadcasting support
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // Check device compatibility
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Div)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Compute broadcasted shape
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;

    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(DivBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                input_ids: [lhs.id(), rhs.id()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }

        return Ok(output);
    }

    // Perform element-wise division based on data type
    let output_data = match result_dtype {
        DataType::Float32 => div_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => div_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 | DataType::Int64 | DataType::Bool => {
            unreachable!("integer and boolean division should coerce to floating point")
        }
    };

    // Create output tensor
    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape.clone(),
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    // Set up gradient function if needed
    if requires_grad {
        let grad_fn = Arc::new(DivBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });

        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Element-wise negation
pub fn neg(tensor: &Tensor) -> Result<Tensor> {
    /// Applies negation for one dtype: fetch the input slice and map
    /// element-wise into a fresh buffer (parallel above `PAR_THRESHOLD`).
    macro_rules! neg_arm {
        ($accessor:ident, $dtype:ident, $tyname:literal) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from tensor"
                ))
            })?;
            TensorData::from_vec(unary_map(input, |i| -i), DataType::$dtype, tensor.device())
        }};
    }

    let output_data = match tensor.dtype() {
        DataType::Float32 => neg_arm!(as_f32_slice, Float32, "f32"),
        DataType::Float64 => neg_arm!(as_f64_slice, Float64, "f64"),
        DataType::Int32 => neg_arm!(as_i32_slice, Int32, "i32"),
        DataType::Int64 => neg_arm!(as_i64_slice, Int64, "i64"),
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "Negation not supported for boolean tensors",
            ));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        let grad_fn = Arc::new(NegBackward {
            input_id: tensor.id(),
        });
        let mut out_with_grad = output;
        out_with_grad.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&out_with_grad, Some(grad_fn))?;
        Ok(out_with_grad)
    } else {
        Ok(output)
    }
}

/// Reject zero divisors for integer floor division / remainder. The integer
/// kernels would otherwise hit a hardware divide-by-zero; floats produce
/// inf/NaN per IEEE and are not checked.
fn ensure_no_integer_zero_divisor(rhs: &Tensor) -> Result<()> {
    let has_zero = match rhs.dtype() {
        DataType::Int32 => rhs
            .data()
            .as_i32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Failed to get i32 divisor slice"))?
            .contains(&0),
        DataType::Int64 => rhs
            .data()
            .as_i64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Failed to get i64 divisor slice"))?
            .contains(&0),
        _ => false,
    };
    if has_zero {
        Err(MinitensorError::invalid_operation(
            "integer floor division or remainder by zero",
        ))
    } else {
        Ok(())
    }
}

/// Element-wise floor division with broadcasting (Python `//` semantics: the
/// quotient rounded toward negative infinity; integer operands stay integral).
///
/// The result never carries a gradient: the derivative is zero almost
/// everywhere and undefined at the jumps, so like PyTorch this op does not
/// participate in autograd.
pub fn floor_div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let (lhs_cast, rhs_cast, result_dtype) =
        coerce_binary_operands(lhs, rhs, BinaryOpKind::FloorDiv)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
    if output_shape.numel() == 0 {
        return Ok(Tensor::empty(
            output_shape,
            result_dtype,
            lhs.device(),
            false,
        ));
    }

    ensure_no_integer_zero_divisor(rhs_ref)?;

    let output_data = match result_dtype {
        DataType::Float32 => floordiv_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => floordiv_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 => floordiv_i32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => floordiv_i64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => unreachable!("bool rejected during operand coercion"),
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        output_shape,
        result_dtype,
        lhs.device(),
        false,
    ))
}

/// Element-wise remainder with broadcasting (Python `%` semantics: the result
/// has the divisor's sign, consistent with [`floor_div`] via
/// `a == floor_div(a, b) * b + remainder(a, b)`).
///
/// Differentiable for float dtypes: `d/dx = 1`, `d/dy = -floor(x/y)`.
pub fn remainder(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Rem)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Gradients only make sense for floating dtypes; an all-integer remainder
    // is exact and non-differentiable.
    let requires_grad = (lhs.requires_grad() || rhs.requires_grad())
        && matches!(result_dtype, DataType::Float32 | DataType::Float64);

    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
    if output_shape.numel() == 0 {
        let mut output = Tensor::empty(
            output_shape.clone(),
            result_dtype,
            lhs.device(),
            requires_grad,
        );
        if requires_grad {
            let grad_fn = Arc::new(RemainderBackward {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                input_ids: [lhs.id(), rhs.id()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            add_to_graph(&output, Some(grad_fn))?;
        }
        return Ok(output);
    }

    ensure_no_integer_zero_divisor(rhs_ref)?;

    let output_data = match result_dtype {
        DataType::Float32 => rem_f32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => rem_f64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int32 => rem_i32_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => rem_i64_direct(lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => unreachable!("bool rejected during operand coercion"),
    };

    let mut output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    if requires_grad {
        let grad_fn = Arc::new(RemainderBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
        });
        output.set_grad_fn(Some(grad_fn.clone()));
        add_to_graph(&output, Some(grad_fn))?;
    }

    Ok(output)
}

/// Element-wise bitwise NOT (`~`): logical NOT for bool tensors, two's
/// complement NOT for integer tensors, rejected for floats — PyTorch's `~`
/// semantics. Non-differentiable by construction.
pub fn bitwise_not(tensor: &Tensor) -> Result<Tensor> {
    /// Applies `!` for one dtype into a fresh buffer, parallel above
    /// `PAR_THRESHOLD`.
    macro_rules! not_arm {
        ($accessor:ident, $dtype:ident, $tyname:literal) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from tensor"
                ))
            })?;
            TensorData::from_vec(unary_map(input, |i| !i), DataType::$dtype, tensor.device())
        }};
    }

    let output_data = match tensor.dtype() {
        DataType::Bool => not_arm!(as_bool_slice, Bool, "bool"),
        DataType::Int32 => not_arm!(as_i32_slice, Int32, "i32"),
        DataType::Int64 => not_arm!(as_i64_slice, Int64, "i64"),
        DataType::Float32 | DataType::Float64 => {
            return Err(MinitensorError::invalid_operation(
                "Bitwise NOT only supported for boolean and integer tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

// Helper functions for type-specific operations
