// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `input @ weight^T + bias`, the affine map behind every dense layer.

use super::matmul_impl::{gemm_nt_f32, gemm_nt_f64};
use crate::{
    autograd::{LinearBackward, add_to_graph},
    error::{MinitensorError, Result},
    ops::arithmetic::add,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// `input @ weight^T + bias`.
///
/// `weight` is `[out_features, in_features]`, the layout PyTorch stores and
/// every checkpoint of this library already uses. Composing this out of
/// `transpose` and `matmul` therefore copies the whole weight matrix on every
/// forward pass -- and then, because the backward differentiates that
/// transpose, copies it again to get `grad_input` and once more to carry the
/// weight gradient back through it. Four full-size copies of a matrix per layer
/// per training step, none of which move any information.
///
/// A GEMM does not need them. Its operands are addressed by row and column
/// stride, so a `[out, in]` matrix *is* a `[in, out]` one with the strides
/// swapped, and `matrixmultiply` packs its operands either way. So the forward
/// takes the weight transposed by stride, and the gradients are two more
/// products of the same operands:
///
/// ```text
/// out        = input @ weight^T
/// grad_input = grad  @ weight            (no transpose at all)
/// grad_weight = grad^T @ input
/// grad_bias  = grad summed over the batch
/// ```
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    if input.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "linear expects an input with at least 2 dimensions",
        ));
    }
    if weight.ndim() != 2 {
        return Err(MinitensorError::invalid_operation(
            "linear expects a 2-dimensional weight",
        ));
    }
    if input.dtype() != weight.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", input.dtype()),
            format!("{:?}", weight.dtype()),
        ));
    }
    if input.device() != weight.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", input.device()),
            format!("{:?}", weight.device()),
        ));
    }
    if !matches!(input.dtype(), DataType::Float32 | DataType::Float64) {
        return Err(MinitensorError::invalid_operation(
            "linear only supports floating point tensors",
        ));
    }

    let dims = input.shape().dims();
    let in_features = dims[dims.len() - 1];
    let out_features = weight.shape().dims()[0];
    if weight.shape().dims()[1] != in_features {
        return Err(MinitensorError::shape_mismatch(
            vec![out_features, in_features],
            weight.shape().dims().to_vec(),
        ));
    }

    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(out_features);
    let out_shape = Shape::new(out_dims);

    let mut output_data =
        TensorData::zeros_on_device(out_shape.numel(), input.dtype(), input.device());

    if rows != 0 && in_features != 0 && out_features != 0 {
        macro_rules! forward {
            ($accessor:ident, $mut_accessor:ident, $gemm:path) => {{
                let a = input.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("linear: unexpected input dtype")
                })?;
                let b = weight.data().$accessor().ok_or_else(|| {
                    MinitensorError::internal_error("linear: unexpected weight dtype")
                })?;
                let c = output_data.$mut_accessor().unwrap();
                // `weight` holds the logical `(in, out)` operand as `(out, in)`.
                unsafe {
                    $gemm(
                        rows,
                        in_features,
                        out_features,
                        a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                    )
                };
            }};
        }
        match input.dtype() {
            DataType::Float32 => forward!(as_f32_slice, as_f32_slice_mut, gemm_nt_f32),
            DataType::Float64 => forward!(as_f64_slice, as_f64_slice_mut, gemm_nt_f64),
            _ => unreachable!("dtype checked above"),
        }
    }

    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.is_some_and(|b| b.requires_grad()));

    let output = Tensor::new(
        Arc::new(output_data),
        out_shape,
        input.dtype(),
        input.device(),
        requires_grad,
    );

    // The bias add is an ordinary broadcast add, so it brings its own gradient
    // and does not need to appear in `LinearBackward`.
    let output = match bias {
        Some(b) => {
            let matmul_part = attach_grad(output, input, weight, requires_grad)?;
            add(&matmul_part, b)?
        }
        None => attach_grad(output, input, weight, requires_grad)?,
    };

    Ok(output)
}

fn attach_grad(
    output: Tensor,
    input: &Tensor,
    weight: &Tensor,
    requires_grad: bool,
) -> Result<Tensor> {
    if !requires_grad {
        return Ok(output);
    }
    let grad_fn = Arc::new(LinearBackward {
        input: input.detach(),
        weight: weight.detach(),
        input_ids: [input.id(), weight.id()],
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.requires_grad(),
    });
    let mut output = output.requires_grad_(true);
    output.set_grad_fn(Some(grad_fn.clone()));
    add_to_graph(&output, Some(grad_fn))?;
    Ok(output)
}
