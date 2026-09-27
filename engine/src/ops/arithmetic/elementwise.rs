// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::autograd::with_grad_fn;
use crate::{
    autograd::{
        AddBackward, DivBackward, MulBackward, NegBackward, RemainderBackward, SubBackward,
    },
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_binary_operands},
    ops::kernels::*,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

pub(crate) use crate::ops::map::PAR_THRESHOLD;
use crate::ops::map::par_out_chunks;
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
    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;

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
            output = with_grad_fn(output, grad_fn)?;
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

        output = with_grad_fn(output, grad_fn)?;
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
        par_out_chunks(lhs, CHUNK, &|start, lhs_chunk| {
            let rhs_chunk = &rhs[start..start + lhs_chunk.len()];
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
    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;

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
            output = with_grad_fn(output, grad_fn)?;
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

        output = with_grad_fn(output, grad_fn)?;
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
    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;

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
            output = with_grad_fn(output, grad_fn)?;
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

        output = with_grad_fn(output, grad_fn)?;
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
    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;

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
            output = with_grad_fn(output, grad_fn)?;
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

        output = with_grad_fn(output, grad_fn)?;
    }

    Ok(output)
}

/// Element-wise negation
pub fn neg(tensor: &Tensor) -> Result<Tensor> {
    /// Applies negation for one dtype: fetch the input slice and map
    /// element-wise into a fresh buffer (parallel above `PAR_THRESHOLD`).
    macro_rules! neg_arm {
        ($accessor:ident, $dtype:ident, $tyname:literal, $negate:expr) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from tensor"
                ))
            })?;
            TensorData::from_vec(unary_map(input, $negate), DataType::$dtype, tensor.device())
        }};
    }

    // The integer arms negate with `wrapping_neg`. `-x` is a panic on
    // `MIN` in a build with overflow checks and a wrap without them, so the
    // same tensor aborted under `cargo test` and returned a value from the
    // released wheel. Two's complement has no representation for `-MIN`, so
    // wrapping is the only answer available; naming it makes the two builds
    // agree on the one the release build was already giving.
    let output_data = match tensor.dtype() {
        DataType::Float32 => neg_arm!(as_f32_slice, Float32, "f32", |v: f32| -v),
        DataType::Float64 => neg_arm!(as_f64_slice, Float64, "f64", |v: f64| -v),
        DataType::Int32 => neg_arm!(as_i32_slice, Int32, "i32", |v: i32| v.wrapping_neg()),
        DataType::Int64 => neg_arm!(as_i64_slice, Int64, "i64", |v: i64| v.wrapping_neg()),
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
        with_grad_fn(output, grad_fn)
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
/// everywhere and undefined at the jumps, so this op does not participate in
/// autograd.
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

    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;
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

/// Which way a modulus rounds its quotient, which is the only thing that
/// separates the two conventions.
#[derive(Clone, Copy)]
enum ModulusConvention {
    /// The quotient rounds towards negative infinity, so the result carries
    /// the *divisor's* sign. Python's `%`, and consistent with [`floor_div`]
    /// via `a == floor_div(a, b) * b + remainder(a, b)`.
    Floored,
    /// The quotient rounds towards zero, so the result carries the
    /// *dividend's* sign. C's `fmod`, and Rust's `%`.
    Truncated,
}

impl ModulusConvention {
    /// The four dtype kernels this convention dispatches to.
    fn kernels(self) -> [BinaryKernel; 4] {
        match self {
            Self::Floored => [
                rem_f32_direct,
                rem_f64_direct,
                rem_i32_direct,
                rem_i64_direct,
            ],
            Self::Truncated => [
                fmod_f32_direct,
                fmod_f64_direct,
                fmod_i32_direct,
                fmod_i64_direct,
            ],
        }
    }

    /// How the backward pass reaches the same quotient. `d/dy` is its
    /// negation, so this is where the two conventions differ in the gradient
    /// as well as in the value.
    fn quotient(self) -> fn(&Tensor, &Tensor) -> Result<Tensor> {
        match self {
            Self::Floored => floor_div,
            Self::Truncated => trunc_div,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Floored => "remainder",
            Self::Truncated => "fmod",
        }
    }

    /// The float64 result from the installed provider, or `None` to compute
    /// it here.
    ///
    /// The engine's float64 modulus is `libm`'s scalar `fmod`, one element at
    /// a time, where NumPy's is a vectorized loop -- 28ns an element against
    /// 18 over 100,000 -- and a modulus is exact, so the answer does not
    /// depend on who computes it. Offered only when each operand is the
    /// output's length or a single element, the two layouts a provider takes.
    fn offer(self, lhs: &Tensor, rhs: &Tensor, len: usize) -> Option<Vec<f64>> {
        use crate::ops::provider::{Ufunc, offer_binary_f64};
        let (left, right) = (lhs.data().as_f64_slice()?, rhs.data().as_f64_slice()?);
        let spans = |values: &[f64]| values.len() == len || values.len() == 1;
        if !(spans(left) && spans(right)) {
            return None;
        }
        let op = match self {
            Self::Floored => Ufunc::Remainder,
            Self::Truncated => Ufunc::Fmod,
        };
        offer_binary_f64(op, left, right, len)
    }
}

/// One dtype's elementwise kernel, as the four in a [`ModulusConvention`] are
/// stored.
type BinaryKernel = fn(&Tensor, &Tensor, &Shape) -> Result<TensorData>;

/// The quotient rounded towards zero, which is what `fmod` subtracts a
/// multiple of. Only ever reached for float operands, since an integer
/// modulus is exact and carries no gradient.
fn trunc_div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    crate::ops::activation::trunc(&div(lhs, rhs)?)
}

/// Element-wise modulus with broadcasting, in either convention.
///
/// Differentiable for float dtypes: `d/dx = 1`, and `d/dy = -q` where `q` is
/// the quotient the convention rounds to, which is locally constant.
fn modulus(lhs: &Tensor, rhs: &Tensor, convention: ModulusConvention) -> Result<Tensor> {
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    let (lhs_cast, rhs_cast, result_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Rem)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();

    // Gradients only make sense for floating dtypes; an all-integer modulus is
    // exact and non-differentiable.
    let requires_grad = (lhs.requires_grad() || rhs.requires_grad())
        && matches!(result_dtype, DataType::Float32 | DataType::Float64);
    let grad_fn = || {
        Arc::new(RemainderBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
            input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
            quotient: convention.quotient(),
        })
    };

    let output_shape = lhs_ref
        .shape()
        .broadcast_with_dtype(rhs_ref.shape(), result_dtype)?;
    if output_shape.numel() == 0 {
        let output = Tensor::empty(output_shape, result_dtype, lhs.device(), requires_grad);
        return if requires_grad {
            with_grad_fn(output, grad_fn())
        } else {
            Ok(output)
        };
    }

    ensure_no_integer_zero_divisor(rhs_ref)?;

    let kernels = convention.kernels();
    let output_data = match result_dtype {
        DataType::Float32 => kernels[0](lhs_ref, rhs_ref, &output_shape)?,
        DataType::Float64 => match convention.offer(lhs_ref, rhs_ref, output_shape.numel()) {
            Some(out) => TensorData::from_vec::<f64>(out, DataType::Float64, lhs.device()),
            None => kernels[1](lhs_ref, rhs_ref, &output_shape)?,
        },
        DataType::Int32 => kernels[2](lhs_ref, rhs_ref, &output_shape)?,
        DataType::Int64 => kernels[3](lhs_ref, rhs_ref, &output_shape)?,
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(format!(
                "{} is not defined for boolean tensors",
                convention.name()
            )));
        }
    };

    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        result_dtype,
        lhs.device(),
        requires_grad,
    );

    if requires_grad {
        return with_grad_fn(output, grad_fn());
    }
    Ok(output)
}

/// Element-wise remainder with broadcasting, carrying the divisor's sign
/// (Python's `%`): `remainder(-7, 3)` is 2.
pub fn remainder(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    modulus(lhs, rhs, ModulusConvention::Floored)
}

/// Element-wise remainder with broadcasting, carrying the dividend's sign
/// (C's `fmod`): `fmod(-7, 3)` is -1.
pub fn fmod(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    modulus(lhs, rhs, ModulusConvention::Truncated)
}

/// The first difference along `dim`: `x[.., i + 1, ..] - x[.., i, ..]`, one
/// shorter along that axis. No gradient is recorded -- a caller that needs
/// one subtracts two narrowed views, which the tape can follow.
///
/// Along an axis every output block is the input block shifted by one row
/// minus the same block unshifted, so each is a single element-wise
/// subtraction of two ranges of one buffer: no copy of either operand, and
/// the same one subtraction per element NumPy's `diff` makes, which gives the
/// same bits. NumPy's runs on one thread; this is the engine's parallel map.
/// Integers wrap, as the engine's `-` does; a bool tensor is refused, as it
/// is there.
pub fn first_difference(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dim >= dims.len() {
        return Err(MinitensorError::invalid_argument(format!(
            "first_difference: dimension {dim} out of range for {} dimensions",
            dims.len()
        )));
    }
    let outer: usize = dims[..dim].iter().product();
    let inner: usize = dims[dim + 1..].iter().product();
    let rows = dims[dim].saturating_sub(1);
    let mut out_dims = dims.to_vec();
    out_dims[dim] = rows;
    let (block_in, block_out) = ((rows + 1) * inner, rows * inner);
    let source = tensor.contiguous()?;

    fn shifted<T>(
        values: &[T],
        outer: usize,
        inner: usize,
        block_in: usize,
        block_out: usize,
        subtract: impl Fn(T, T) -> T + Send + Sync + Copy,
    ) -> Vec<T>
    where
        T: Copy + Send + Sync,
    {
        if block_out == 0 {
            return Vec::new();
        }
        if outer == 1 {
            return crate::ops::map::binary_map(
                &values[inner..inner + block_out],
                &values[..block_out],
                subtract,
            );
        }
        // SAFETY: the chunks tile the output in whole blocks, and every block
        // is written in full by `zip_slices_into`, which walks the whole zip
        // of three slices of the block's length.
        unsafe {
            crate::ops::map::build_vec(outer * block_out, |spare| {
                crate::ops::map::par_out_chunks_sized(
                    spare,
                    block_out,
                    std::mem::size_of_val(values),
                    &|start, piece: &mut [std::mem::MaybeUninit<T>]| {
                        for (offset, block) in piece.chunks_mut(block_out).enumerate() {
                            let base = (start / block_out + offset) * block_in;
                            crate::ops::map::zip_slices_into(
                                &values[base + inner..base + inner + block_out],
                                &values[base..base + block_out],
                                block,
                                &subtract,
                            );
                        }
                    },
                )
            })
        }
    }

    let device = tensor.device();
    let data = source.data();
    macro_rules! arm {
        ($accessor:ident, $from_vec:ident, $subtract:expr) => {{
            let values = data.$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "first_difference: failed to read the input with ",
                    stringify!($accessor)
                ))
            })?;
            TensorData::$from_vec(
                shifted(values, outer, inner, block_in, block_out, $subtract),
                device,
            )
        }};
    }
    let output_data = match tensor.dtype() {
        DataType::Float32 => arm!(as_f32_slice, from_vec_f32, |a: f32, b: f32| a - b),
        DataType::Float64 => arm!(as_f64_slice, from_vec_f64, |a: f64, b: f64| a - b),
        DataType::Int32 => arm!(as_i32_slice, from_vec_i32, |a: i32, b: i32| a
            .wrapping_sub(b)),
        DataType::Int64 => arm!(as_i64_slice, from_vec_i64, |a: i64, b: i64| a
            .wrapping_sub(b)),
        DataType::Bool => {
            return Err(MinitensorError::invalid_operation(
                "first_difference is not defined for boolean tensors",
            ));
        }
    };
    Ok(Tensor::new(
        Arc::new(output_data),
        Shape::new(out_dims),
        tensor.dtype(),
        device,
        false,
    ))
}

// Helper functions for type-specific operations

#[cfg(test)]
mod modulus_tests {
    use super::*;
    use crate::{autograd::backward_collect, device::Device, tensor::Shape};

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn i64_tensor(data: Vec<i64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_i64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Int64,
            Device::cpu(),
            false,
        )
    }

    fn wide(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    /// `1.0 / 0.1` rounds up onto 10.0 from just below it, so the floor of
    /// the rounded quotient is one too many; the floored quotient is 9, and
    /// with it `q * b + remainder(a, b)` gives back `a`.
    #[test]
    fn a_quotient_rounded_onto_an_integer_is_floored_below_it() {
        let a = f64_tensor(vec![1.0, 0.3, 2.0, -1.0, 6.0]);
        let b = f64_tensor(vec![0.1, 0.1, 0.2, 0.1, 3.0]);
        assert_eq!(
            wide(&floor_div(&a, &b).unwrap()),
            vec![9.0, 2.0, 9.0, -10.0, 2.0]
        );
        let q = wide(&floor_div(&a, &b).unwrap());
        let r = wide(&remainder(&a, &b).unwrap());
        for (i, (&dividend, &divisor)) in wide(&a).iter().zip(&wide(&b)).enumerate() {
            assert!((q[i] * divisor + r[i] - dividend).abs() < 1e-15, "{i}");
        }
    }

    /// The first difference along each axis of a 2x3x4 block against the
    /// definition, element by element: the one-block path (axis 0 of a
    /// vector-like leading axis) and the per-block one alike, integers
    /// wrapping at the end of their range, and bool refused.
    #[test]
    fn the_first_difference_is_the_next_row_minus_this_one() {
        let dims = [2usize, 3, 4];
        let values: Vec<f64> = (0..24).map(|v| (v * v) as f64 * 0.5).collect();
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f64(values.clone(), Device::cpu())),
            Shape::new(dims.to_vec()),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let strides = [12usize, 4, 1];
        for axis in 0..3 {
            let got = first_difference(&tensor, axis).unwrap();
            let mut out_dims = dims;
            out_dims[axis] -= 1;
            assert_eq!(got.shape().dims(), &out_dims);
            let got = wide(&got);
            let mut k = 0;
            for i in 0..out_dims[0] {
                for j in 0..out_dims[1] {
                    for l in 0..out_dims[2] {
                        let at = i * strides[0] + j * strides[1] + l * strides[2];
                        assert_eq!(got[k], values[at + strides[axis]] - values[at]);
                        k += 1;
                    }
                }
            }
        }

        let ends = i64_tensor(vec![i64::MIN, 1]);
        let wrapped = first_difference(&ends, 0).unwrap();
        assert_eq!(
            wrapped.data().as_i64_slice().unwrap(),
            &[1i64.wrapping_sub(i64::MIN)]
        );

        let truth = Tensor::new(
            Arc::new(TensorData::from_vec_bool(vec![true, false], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Bool,
            Device::cpu(),
            false,
        );
        assert!(first_difference(&truth, 0).is_err());
        assert!(first_difference(&ends, 1).is_err());
    }

    /// A remainder of zero takes the divisor's sign, as Python's `%` and
    /// NumPy's `remainder` give it, while `fmod`'s zero keeps the dividend's.
    /// Both widths, since only float64 is ever handed to a provider: these
    /// run on a bare engine with none installed, which is the engine's own
    /// kernel either way.
    #[test]
    fn a_zero_remainder_takes_the_divisors_sign() {
        let dividends = [-4.0, 4.0, -0.0, 0.0, 6.0, -6.0];
        let divisors = [2.0, -2.0, 3.0, -3.0, 3.0, -3.0];
        // (remainder, fmod) signs: negative?
        let remainder_negative = [false, true, false, true, false, true];
        let fmod_negative = [true, false, true, false, false, true];

        let (a, b) = (
            f64_tensor(dividends.to_vec()),
            f64_tensor(divisors.to_vec()),
        );
        let got = wide(&remainder(&a, &b).unwrap());
        let truncated = wide(&fmod(&a, &b).unwrap());
        for i in 0..dividends.len() {
            assert_eq!(got[i], 0.0);
            assert_eq!(
                got[i].is_sign_negative(),
                remainder_negative[i],
                "remainder {i}"
            );
            assert_eq!(
                truncated[i].is_sign_negative(),
                fmod_negative[i],
                "fmod {i}"
            );
        }

        let narrow = |values: &[f64]| {
            Tensor::new(
                Arc::new(TensorData::from_vec_f32(
                    values.iter().map(|&v| v as f32).collect(),
                    Device::cpu(),
                )),
                Shape::new(vec![values.len()]),
                DataType::Float32,
                Device::cpu(),
                false,
            )
        };
        let got = remainder(&narrow(&dividends), &narrow(&divisors)).unwrap();
        for (i, value) in got.data().as_f32_slice().unwrap().iter().enumerate() {
            assert_eq!(
                value.is_sign_negative(),
                remainder_negative[i],
                "f32 remainder {i}"
            );
        }
    }

    /// Every sign pairing, which is the only place the two conventions differ.
    const DIVIDENDS: [f64; 4] = [7.0, -7.0, 7.0, -7.0];
    const DIVISORS: [f64; 4] = [3.0, 3.0, -3.0, -3.0];

    #[test]
    fn the_two_conventions_differ_only_when_the_signs_disagree() {
        let a = f64_tensor(DIVIDENDS.to_vec());
        let b = f64_tensor(DIVISORS.to_vec());

        // `remainder` takes the divisor's sign, `fmod` the dividend's.
        assert_eq!(
            wide(&remainder(&a, &b).unwrap()),
            vec![1.0, 2.0, -2.0, -1.0]
        );
        assert_eq!(wide(&fmod(&a, &b).unwrap()), vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn each_convention_reconstructs_its_own_quotient() {
        // The identity that defines them: `a == q * b + r`, with `q` floored
        // for one and truncated for the other.
        let a = f64_tensor(DIVIDENDS.to_vec());
        let b = f64_tensor(DIVISORS.to_vec());

        for (name, values, quotients) in [
            (
                "remainder",
                wide(&remainder(&a, &b).unwrap()),
                DIVIDENDS
                    .iter()
                    .zip(DIVISORS)
                    .map(|(x, y)| (x / y).floor())
                    .collect::<Vec<_>>(),
            ),
            (
                "fmod",
                wide(&fmod(&a, &b).unwrap()),
                DIVIDENDS
                    .iter()
                    .zip(DIVISORS)
                    .map(|(x, y)| (x / y).trunc())
                    .collect::<Vec<_>>(),
            ),
        ] {
            for (index, ((&r, &q), (&x, y))) in values
                .iter()
                .zip(&quotients)
                .zip(DIVIDENDS.iter().zip(DIVISORS))
                .enumerate()
            {
                assert_eq!(q * y + r, x, "{name}[{index}]");
            }
        }
    }

    #[test]
    fn integer_operands_stay_integral_in_both() {
        let a = i64_tensor(vec![7, -7, 7, -7]);
        let b = i64_tensor(vec![3, 3, -3, -3]);
        for (name, result) in [("remainder", remainder(&a, &b)), ("fmod", fmod(&a, &b))] {
            let out = result.unwrap();
            assert_eq!(out.dtype(), DataType::Int64, "{name}");
        }
        assert_eq!(
            remainder(&a, &b).unwrap().data().as_i64_slice().unwrap(),
            &[1, 2, -2, -1]
        );
        assert_eq!(
            fmod(&a, &b).unwrap().data().as_i64_slice().unwrap(),
            &[1, -1, 1, -1]
        );
    }

    #[test]
    fn an_integer_zero_divisor_is_refused_by_both() {
        let a = i64_tensor(vec![1]);
        let zero = i64_tensor(vec![0]);
        assert!(remainder(&a, &zero).is_err());
        assert!(fmod(&a, &zero).is_err());
    }

    #[test]
    fn the_gradient_follows_each_convention_s_own_quotient() {
        // `d/dx` is 1 for both; `d/dy` is the negated quotient, which is where
        // they part company for mixed signs.
        for (name, op, expected) in [
            (
                "remainder",
                remainder as fn(&Tensor, &Tensor) -> Result<Tensor>,
                // floor(-7/3) = -3
                3.0,
            ),
            (
                "fmod", fmod, // trunc(-7/3) = -2
                2.0,
            ),
        ] {
            let a = f64_tensor(vec![-7.0]).requires_grad_(true);
            let b = f64_tensor(vec![3.0]).requires_grad_(true);
            let out = op(&a, &b).unwrap();
            let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
            let grads = backward_collect(&out, Some(seed)).unwrap();
            assert_eq!(wide(grads.get(&a.id()).unwrap()), vec![1.0], "{name} d/dx");
            assert_eq!(
                wide(grads.get(&b.id()).unwrap()),
                vec![expected],
                "{name} d/dy"
            );
        }
    }

    #[test]
    fn an_integer_modulus_carries_no_gradient() {
        let a = i64_tensor(vec![7]);
        let b = i64_tensor(vec![3]);
        assert!(!fmod(&a, &b).unwrap().requires_grad());
        assert!(!remainder(&a, &b).unwrap().requires_grad());
    }
}
