// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `input @ weight^T + bias`, the affine map behind every dense layer.

use crate::autograd::with_grad_fn;
use crate::{
    autograd::LinearBackward,
    error::{MinitensorError, Result},
    ops::arithmetic::add,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// `input @ weight^T + bias`.
///
/// `weight` is `[out_features, in_features]`, the layout every checkpoint of
/// this library already uses. Composing this out of
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
        TensorData::try_zeros_on_device(out_shape.numel(), input.dtype(), input.device())?;

    // A bias of the input's dtype holding one value per output feature --
    // `[out]`, or that with leading ones no wider than the output -- is added
    // in the product's own kernel: for float32 each entry is then the float32
    // nearest `input · weight + bias` exactly, where adding the bias to the
    // rounded product rounded twice. Any other bias broadcasts as `+` does.
    let fused_bias = bias.filter(|b| {
        let dims = b.shape().dims();
        b.dtype() == input.dtype()
            && b.device() == input.device()
            && dims.len() <= out_shape.dims().len()
            && dims.last() == Some(&out_features)
            && dims[..dims.len() - 1].iter().all(|&d| d == 1)
    });
    let fused_values = fused_bias.map(Tensor::contiguous).transpose()?;

    if rows != 0 && out_features != 0 && (in_features != 0 || fused_bias.is_some()) {
        // Each entry the float nearest its exact dot, the bias added to it
        // exactly; `weight` holds the logical `(in, out)` operand as `(out,
        // in)` and is read that way round.
        use crate::ops::provider::Storage;
        let taken = fused_values
            .as_ref()
            .map(|values| crate::ops::linalg::Taken {
                alpha: 1.0,
                beta: 1.0,
                addend: Some((&**values.data(), [0, 0, 1])),
            });
        crate::ops::linalg::rounded_products(
            (1, rows, in_features, out_features),
            (&**input.data(), Storage::RowMajor),
            (&**weight.data(), Storage::Transposed),
            taken,
            &mut output_data,
        )?;
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

    // A bias the kernel did not take is an ordinary broadcast add, which
    // brings its own gradient; one it took has its gradient in
    // `LinearBackward`.
    let output = match (bias, fused_bias) {
        (Some(b), None) => {
            let matmul_part = attach_grad(output, input, weight, None, requires_grad)?;
            add(&matmul_part, b)?
        }
        _ => attach_grad(output, input, weight, fused_bias, requires_grad)?,
    };

    Ok(output)
}

fn attach_grad(
    output: Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    requires_grad: bool,
) -> Result<Tensor> {
    if !requires_grad {
        return Ok(output);
    }
    let mut input_ids = smallvec::smallvec![input.id(), weight.id()];
    let bias = bias.filter(|b| b.requires_grad());
    if let Some(b) = bias {
        input_ids.push(b.id());
    }
    let grad_fn = Arc::new(LinearBackward {
        input: input.detach(),
        weight: weight.detach(),
        input_ids,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.requires_grad(),
        bias_shape: bias.map(|b| b.shape().clone()),
    });
    let output = output.requires_grad_(true);
    with_grad_fn(output, grad_fn)
}

/// `beta · input + alpha · (mat1 @ mat2)` for float operands of one width,
/// each entry the float nearest its exact value, with its gradient: `addmm`
/// for two matrices and `baddbmm` for two stacks of them of one depth,
/// `input` broadcasting to the product's shape. The scales are floats of
/// the operands' width.
///
/// Composed of a product, two scalings and a sum, an entry was rounded at
/// each; here the product's exact dot is scaled and the addend added to it
/// before the one rounding. See [`crate::ops::linalg::Affine`].
pub fn affine_product(
    input: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    beta: f64,
    alpha: f64,
) -> Result<Tensor> {
    use crate::ops::provider::Storage;
    let dtype = input.dtype();
    if !matches!(dtype, DataType::Float32 | DataType::Float64)
        || [mat1, mat2]
            .iter()
            .any(|t| t.dtype() != dtype || t.device() != input.device())
    {
        return Err(MinitensorError::invalid_operation(
            "an affine product takes float operands of one width on one device",
        ));
    }
    let (left, right) = (mat1.shape().dims(), mat2.shape().dims());
    let batched = left.len() == 3;
    if !(left.len() == 2 || batched)
        || right.len() != left.len()
        || (batched && left[0] != right[0])
        || left[left.len() - 1] != right[right.len() - 2]
    {
        return Err(MinitensorError::shape_mismatch(
            left.to_vec(),
            right.to_vec(),
        ));
    }
    let batch = if batched { left[0] } else { 1 };
    let (m, k, n) = (
        left[left.len() - 2],
        left[left.len() - 1],
        right[right.len() - 1],
    );
    let out_dims: Vec<usize> = if batched {
        vec![batch, m, n]
    } else {
        vec![m, n]
    };
    // The addend's stride along the batch, its rows and its columns: its own
    // where it has the axis at full length, and zero where it broadcasts.
    let base = input.contiguous()?;
    let base_dims = base.shape().dims();
    if base_dims.len() > out_dims.len() {
        return Err(MinitensorError::shape_mismatch(
            out_dims.clone(),
            base_dims.to_vec(),
        ));
    }
    let mut full = [1usize; 3];
    full[3 - base_dims.len()..].copy_from_slice(base_dims);
    let target = [batch, m, n];
    let mut strides = [0usize; 3];
    let mut step = 1;
    for axis in (0..3).rev() {
        if full[axis] == target[axis] && target[axis] != 1 {
            strides[axis] = step;
        } else if full[axis] != 1 {
            return Err(MinitensorError::shape_mismatch(
                out_dims.clone(),
                base_dims.to_vec(),
            ));
        }
        step *= full[axis];
    }
    let (lhs, rhs) = (mat1.contiguous()?, mat2.contiguous()?);
    let mut data = TensorData::try_zeros_on_device(batch * m * n, dtype, input.device())?;
    crate::ops::linalg::rounded_products(
        (batch, m, k, n),
        (&**lhs.data(), Storage::RowMajor),
        (&**rhs.data(), Storage::RowMajor),
        Some(crate::ops::linalg::Taken {
            alpha,
            beta,
            addend: Some((&**base.data(), strides)),
        }),
        &mut data,
    )?;
    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad() || mat1.requires_grad() || mat2.requires_grad());
    let output = Tensor::new(
        Arc::new(data),
        Shape::new(out_dims),
        dtype,
        input.device(),
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }
    let grad_fn = Arc::new(crate::autograd::AffineProductBackward {
        mat1: lhs.detach(),
        mat2: rhs.detach(),
        input_shape: input.shape().clone(),
        alpha,
        beta,
        input_ids: [input.id(), mat1.id(), mat2.id()],
        requires: [
            input.requires_grad(),
            mat1.requires_grad(),
            mat2.requires_grad(),
        ],
    });
    with_grad_fn(output, grad_fn)
}
