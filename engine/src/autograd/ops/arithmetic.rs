// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::{
    error::{MinitensorError, Result},
    ops::map::{binary_map, outputs_per_task, par_out_chunks, ternary_map, unary_map},
    ops::util::create_scalar_tensor,
    ops::{arithmetic, reduction, shape_ops},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

pub(crate) fn expand_reduction_grad(
    grad_output: &Tensor,
    input_shape: &[usize],
    dims: &Option<Vec<usize>>,
    keepdim: bool,
) -> Result<Tensor> {
    if keepdim {
        return Ok(grad_output.clone());
    }

    if let Some(dims) = dims {
        let mut shape = grad_output.shape().dims().to_vec();
        let mut sorted = dims.clone();
        sorted.sort_unstable();
        for &d in &sorted {
            shape.insert(d, 1);
        }
        shape_ops::reshape(grad_output, Shape::new(shape))
    } else {
        shape_ops::reshape(grad_output, Shape::new(vec![1; input_shape.len()]))
    }
}

impl GradientFunction for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad = expand_reduction_grad(grad_output, &self.input_shape, &self.dims, self.keepdim)?;

        let ones = Tensor::ones(
            Shape::new(self.input_shape.clone()),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        let grad_input = arithmetic::mul(&ones, &grad)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for NaN-aware sum reduction
pub struct NanSumBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
    pub mask: Tensor,
}

impl GradientFunction for NanSumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad = expand_reduction_grad(grad_output, &self.input_shape, &self.dims, self.keepdim)?;
        let mask = self.mask.astype(grad_output.dtype())?;
        let grad_input = arithmetic::mul(&mask, &grad)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for NaN-aware mean reduction
pub struct NanMeanBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
    pub mask: Tensor,
    pub count: Tensor,
}

impl GradientFunction for NanMeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad = expand_reduction_grad(grad_output, &self.input_shape, &self.dims, self.keepdim)?;
        let count =
            expand_reduction_grad(&self.count, &self.input_shape, &self.dims, self.keepdim)?;
        let grad = sanitize_grad_for_nanmean(&grad, &count)?;
        let count = safe_count_for_nanmean(&count)?;

        let scaled = arithmetic::div(&grad, &count)?;
        let mask = self.mask.astype(grad_output.dtype())?;
        let grad_input = arithmetic::mul(&mask, &scaled)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

fn sanitize_grad_for_nanmean(grad: &Tensor, count: &Tensor) -> Result<Tensor> {
    if grad.dtype() != count.dtype() {
        return Err(MinitensorError::invalid_operation(
            "nanmean backward expected matching gradient and count dtypes",
        ));
    }

    // A count of zero means the whole reduced axis was NaN, so nothing flows
    // back through that position.
    macro_rules! sanitize {
        ($accessor:ident, $ty:ty, $tyname:literal, $from_vec:ident) => {{
            let missing =
                || MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"));
            let grad_src = grad.data().$accessor().ok_or_else(missing)?;
            let count_src = count.data().$accessor().ok_or_else(missing)?;
            TensorData::$from_vec(
                binary_map(
                    grad_src,
                    count_src,
                    |g: $ty, c: $ty| {
                        if c == 0.0 { 0.0 } else { g }
                    },
                ),
                grad.device(),
            )
        }};
    }

    let new_data = match grad.dtype() {
        DataType::Float32 => sanitize!(as_f32_slice, f32, "f32", from_vec_f32),
        DataType::Float64 => sanitize!(as_f64_slice, f64, "f64", from_vec_f64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmean backward only supports floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(new_data),
        grad.shape().clone(),
        grad.dtype(),
        grad.device(),
        false,
    ))
}

fn safe_count_for_nanmean(count: &Tensor) -> Result<Tensor> {
    // Divide-by-zero guard: a zero count pairs with a zero gradient (see
    // `sanitize_grad_for_nanmean`), so the substituted 1 never reaches a result.
    macro_rules! safe_count {
        ($accessor:ident, $ty:ty, $tyname:literal, $from_vec:ident) => {{
            let src = count.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!("Failed to get ", $tyname, " slice"))
            })?;
            TensorData::$from_vec(
                unary_map(src, |c: $ty| if c == 0.0 { 1.0 } else { c }),
                count.device(),
            )
        }};
    }

    let new_data = match count.dtype() {
        DataType::Float32 => safe_count!(as_f32_slice, f32, "f32", from_vec_f32),
        DataType::Float64 => safe_count!(as_f64_slice, f64, "f64", from_vec_f64),
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmean backward only supports floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(new_data),
        count.shape().clone(),
        count.dtype(),
        count.device(),
        false,
    ))
}

/// Gradient function for product reduction
pub struct ProdBackward {
    pub input: Tensor,
    pub result: Tensor,
    pub input_id: TensorId,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
}

impl GradientFunction for ProdBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let input = &self.input;
        let input_shape = input.shape().dims().to_vec();
        let dtype = input.dtype();
        let device = input.device();

        // Broadcast the upstream gradient back over the reduced axes.
        let grad = expand_reduction_grad(grad_output, &input_shape, &self.dims, self.keepdim)?;

        let reduce_dims: Option<Vec<isize>> = self
            .dims
            .as_ref()
            .map(|dims| dims.iter().map(|&d| d as isize).collect());

        // d(prod)/dx_i is the product of the *other* elements in the reduction
        // group. Computing it as `total_product / x_i` breaks when the group
        // contains zeros (0 / 0 = NaN), so handle zeros explicitly:
        //   - no zeros in the group:  grad_i = P / x_i
        //   - exactly one zero:       grad_i = product of the non-zero elements
        //                             at the zero position, 0 elsewhere
        //   - two or more zeros:      grad_i = 0 everywhere
        let zero = create_scalar_tensor(0.0, dtype, device)?;
        let is_zero = crate::ops::comparison::eq(input, &zero)?; // bool mask
        let is_zero_f = is_zero.astype(dtype)?;
        let ones = Tensor::ones(input.shape().clone(), dtype, device, false);

        // Per-group zero count and product of the non-zero elements.
        let zero_count = reduction::sum(&is_zero_f, reduce_dims.clone(), true)?;
        let safe_input = crate::ops::selection::where_op(&is_zero, &ones, input)?;
        let prod_nonzero = reduction::prod(&safe_input, reduce_dims, true)?;

        let one_scalar = create_scalar_tensor(1.0, dtype, device)?;
        let no_zero = crate::ops::comparison::eq(&zero_count, &zero)?.astype(dtype)?;
        let one_zero = crate::ops::comparison::eq(&zero_count, &one_scalar)?.astype(dtype)?;

        // Contribution at the (unique) zero position: product of the others.
        let zero_term = arithmetic::mul(&is_zero_f, &one_zero)?;
        let zero_term = arithmetic::mul(&zero_term, &prod_nonzero)?;

        // Contribution at non-zero positions when the group has no zeros.
        let nonzero_mask = arithmetic::sub(&ones, &is_zero_f)?;
        let quotient = arithmetic::div(&prod_nonzero, &safe_input)?;
        let nonzero_term = arithmetic::mul(&nonzero_mask, &no_zero)?;
        let nonzero_term = arithmetic::mul(&nonzero_term, &quotient)?;

        let per_element = arithmetic::add(&zero_term, &nonzero_term)?;
        let grad_input = arithmetic::mul(&grad, &per_element)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for cumulative sum operation
pub struct CumsumBackward {
    pub input_id: TensorId,
    pub dim: usize,
}

/// Gradient function for cumulative product operation
pub struct CumprodBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for CumprodBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad_input =
            reduction::cumprod_backward(&self.input, &self.output, grad_output, self.dim)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

impl GradientFunction for CumsumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad_input = reduction::cumsum_backward(grad_output, self.dim)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

// Gradient functions for activation functions

/// Gradient function for exponential
pub struct ExpBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(exp(x)) = exp(x) * grad_output
        let grad = arithmetic::mul(&self.output, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for square root.
///
/// Saves the output `y = sqrt(x)` (rather than the input) so the derivative
/// `d/dx sqrt(x) = 0.5 / sqrt(x)` is a single division: `grad_output / (2y)`.
/// At `x == 0` this divides by zero and yields `inf`.
pub struct SqrtBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for SqrtBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx sqrt(x) = 0.5 / sqrt(x) = 1 / (2 * y)
        let two_y = arithmetic::add(&self.output, &self.output)?;
        let grad = arithmetic::div(grad_output, &two_y)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for reciprocal square root.
///
/// Saves the output `y = x^(-1/2)` so the derivative
/// `d/dx x^(-1/2) = -0.5 * x^(-3/2) = -0.5 * y^3` needs no reference to the
/// input.
pub struct RsqrtBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for RsqrtBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx rsqrt(x) = -0.5 * y^3
        let y2 = arithmetic::mul(&self.output, &self.output)?;
        let y3 = arithmetic::mul(&y2, &self.output)?;
        let scaled = arithmetic::mul(&y3, grad_output)?;
        let neg_half = create_scalar_tensor(-0.5, self.output.dtype(), self.output.device())?;
        let grad = arithmetic::mul(&scaled, &neg_half)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for logarithm
pub struct LogBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(log(x)) = 1/x * grad_output
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let inv = arithmetic::div(&ones, &self.input.detach())?;
        let grad = arithmetic::mul(&inv, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for log1p
pub struct Log1pBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for Log1pBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::add(&ones, &self.input.detach())?;
        let grad = arithmetic::div(grad_output, &denom)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for expm1
pub struct Expm1Backward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for Expm1Backward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::add(&self.output.detach(), &ones)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for sine
pub struct SinBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SinBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(sin(x)) = cos(x) * grad_output
        let cos_x = self.input.cos()?;
        let grad = arithmetic::mul(&cos_x, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for cosine
pub struct CosBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for CosBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(cos(x)) = -sin(x) * grad_output
        let sin_x = self.input.sin()?;
        let mul = arithmetic::mul(&sin_x, grad_output)?;
        let grad = arithmetic::neg(&mul)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for tangent
pub struct TanBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(tan(x)) = (1 + tan²(x)) * grad_output
        let tan_sq = arithmetic::mul(&self.output, &self.output)?;
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::add(&ones, &tan_sq)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse sine
pub struct AsinBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AsinBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(asin(x)) = grad_output / sqrt(1 - x^2)
        let square = arithmetic::mul(&self.input, &self.input)?;
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::sub(&ones, &square)?;
        let sqrt = denom.sqrt()?;
        let grad = arithmetic::div(grad_output, &sqrt)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse cosine
pub struct AcosBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AcosBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(acos(x)) = -grad_output / sqrt(1 - x^2)
        let square = arithmetic::mul(&self.input, &self.input)?;
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::sub(&ones, &square)?;
        let sqrt = denom.sqrt()?;
        let frac = arithmetic::div(grad_output, &sqrt)?;
        let grad = arithmetic::neg(&frac)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse tangent
pub struct AtanBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AtanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(atan(x)) = grad_output / (1 + x^2)
        let square = arithmetic::mul(&self.input, &self.input)?;
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::add(&ones, &square)?;
        let grad = arithmetic::div(grad_output, &denom)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for hyperbolic sine
pub struct SinhBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SinhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(sinh(x)) = cosh(x) * grad_output
        let cosh_x = self.input.cosh()?;
        let grad = arithmetic::mul(&cosh_x, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for hyperbolic cosine
pub struct CoshBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for CoshBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(cosh(x)) = sinh(x) * grad_output
        let sinh_x = self.input.sinh()?;
        let grad = arithmetic::mul(&sinh_x, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse hyperbolic sine
pub struct AsinhBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AsinhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(asinh(x)) = grad_output / sqrt(1 + x^2)
        let square = arithmetic::mul(&self.input, &self.input)?;
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::add(&square, &ones)?;
        let sqrt = denom.sqrt()?;
        let grad = arithmetic::div(grad_output, &sqrt)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse hyperbolic cosine
pub struct AcoshBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AcoshBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(acosh(x)) = grad_output / sqrt((x - 1)(x + 1))
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let x_minus_one = arithmetic::sub(&self.input, &ones)?;
        let x_plus_one = arithmetic::add(&self.input, &ones)?;
        let product = arithmetic::mul(&x_minus_one, &x_plus_one)?;
        let sqrt = product.sqrt()?;
        let grad = arithmetic::div(grad_output, &sqrt)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for inverse hyperbolic tangent
pub struct AtanhBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for AtanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(atanh(x)) = grad_output / (1 - x^2)
        let square = arithmetic::mul(&self.input, &self.input)?;
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::sub(&ones, &square)?;
        let grad = arithmetic::div(grad_output, &denom)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `log2` and `log10`.
///
/// `d/dx log_b(x) = 1 / (x * ln b)`, so the two bases differ only by the stored
/// constant and share one implementation.
pub struct LogBaseBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    /// `ln(base)`: `std::f64::consts::LN_2` or `LN_10`.
    pub ln_base: f64,
}

impl GradientFunction for LogBaseBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let ln_base = self.ln_base;
        let grad = unary_chain_grad_f64(&self.input, grad_output, "log_base", move |x| {
            1.0 / (x * ln_base)
        })?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `erf` and `erfc`.
///
/// `d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)`, and `erfc = 1 - erf` so its
/// derivative is the same thing negated — `scale` carries the sign.
pub struct ErfBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    /// `+2/sqrt(pi)` for `erf`, `-2/sqrt(pi)` for `erfc`.
    pub scale: f64,
}

impl GradientFunction for ErfBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let scale = self.scale;
        let grad = unary_chain_grad_f64(&self.input, grad_output, "erf", move |x| {
            scale * (-x * x).exp()
        })?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for sum reduction
pub struct SumBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
}
/// Gradient function for logaddexp
pub struct LogAddExpBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub input_shapes: [Vec<usize>; 2],
    /// Which inputs actually need a gradient; frozen inputs skip their
    /// exp/sub/mul/reduce chain entirely.
    pub input_requires_grad: [bool; 2],
}
impl GradientFunction for LogAddExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        if self.input_requires_grad[0] {
            let lhs_diff = arithmetic::sub(&self.lhs.detach(), &self.output.detach())?;
            let lhs_term = lhs_diff.exp()?;
            let lhs_mul = arithmetic::mul(&lhs_term, grad_output)?;
            let lhs_grad = reduce_gradient_for_broadcasting(
                &lhs_mul,
                &Shape::new(self.input_shapes[0].clone()),
            )?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        if self.input_requires_grad[1] {
            let rhs_diff = arithmetic::sub(&self.rhs.detach(), &self.output.detach())?;
            let rhs_term = rhs_diff.exp()?;
            let rhs_mul = arithmetic::mul(&rhs_term, grad_output)?;
            let rhs_grad = reduce_gradient_for_broadcasting(
                &rhs_mul,
                &Shape::new(self.input_shapes[1].clone()),
            )?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for nan_to_num.
pub struct NanToNumBackward {
    pub input_id: TensorId,
    pub finite_mask: Vec<bool>,
}
impl GradientFunction for NanToNumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        if self.finite_mask.len() != grad_output.numel() {
            return Err(MinitensorError::gradient_error(
                "nan_to_num backward mask length does not match gradient size",
            ));
        }

        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let grad = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let out = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                apply_finite_mask(grad, out, &self.finite_mask);
            }
            DataType::Float64 => {
                let grad = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let out = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                apply_finite_mask(grad, out, &self.finite_mask);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "nan_to_num backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
#[inline(always)]
fn apply_finite_mask<T>(grad: &[T], output: &mut [T], finite_mask: &[bool])
where
    T: Copy + Default + Send + Sync,
{
    debug_assert_eq!(grad.len(), output.len());
    debug_assert_eq!(grad.len(), finite_mask.len());

    let len = grad.len();
    if len < PAR_THRESHOLD {
        for i in 0..len {
            output[i] = if finite_mask[i] {
                grad[i]
            } else {
                T::default()
            };
        }
    } else {
        par_out_chunks(output, outputs_per_task(1), &|start, chunk| {
            for (offset, out) in chunk.iter_mut().enumerate() {
                let i = start + offset;
                *out = if finite_mask[i] {
                    grad[i]
                } else {
                    T::default()
                };
            }
        });
    }
}
/// Gradient function for the element-wise absolute value.
///
/// `d/dx |x| = sign(x)` with the sub-gradient at `x == 0` taken as `0`.
/// The stored input shares storage with the forward input (a detached
/// clone), so no data is copied.
pub struct AbsBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}
impl GradientFunction for AbsBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        macro_rules! abs_grad {
            ($slice:ident, $ty:ty, $from_vec:ident) => {{
                let x = self.input.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read input for abs backward")
                })?;
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for abs backward")
                })?;
                TensorData::$from_vec(
                    binary_map(go, x, |o: $ty, v: $ty| {
                        let sign: $ty = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        o * sign
                    }),
                    grad_output.device(),
                )
            }};
        }

        let grad_data = match grad_output.dtype() {
            DataType::Float32 => abs_grad!(as_f32_slice, f32, from_vec_f32),
            DataType::Float64 => abs_grad!(as_f64_slice, f64, from_vec_f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "abs backward only supported for floating point tensors",
                ));
            }
        };

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for `clamp`/`clip`.
///
/// The gradient is passed through where the input lies inside the (inclusive)
/// clamp bounds and zeroed where it was saturated. Either
/// bound may be absent (`clamp_min`/`clamp_max`).
pub struct ClampBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub min: Option<f64>,
    pub max: Option<f64>,
}
impl GradientFunction for ClampBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        macro_rules! clamp_grad {
            ($slice:ident, $ty:ty, $from_vec:ident) => {{
                let x = self.input.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read input for clamp backward")
                })?;
                let go = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read grad_output for clamp backward")
                })?;
                let min = self.min.map(|m| m as $ty);
                let max = self.max.map(|m| m as $ty);
                TensorData::$from_vec(
                    binary_map(go, x, move |o: $ty, v: $ty| {
                        let passes = min.map_or(true, |m| v >= m) && max.map_or(true, |m| v <= m);
                        if passes { o } else { 0.0 }
                    }),
                    grad_output.device(),
                )
            }};
        }

        let grad_data = match grad_output.dtype() {
            DataType::Float32 => clamp_grad!(as_f32_slice, f32, from_vec_f32),
            DataType::Float64 => clamp_grad!(as_f64_slice, f64, from_vec_f64),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "clamp backward only supported for floating point tensors",
                ));
            }
        };

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
/// Gradient function for power operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowBroadcast {
    None,
    BaseScalar,
    ExponentScalar,
}
pub struct PowBackward {
    pub base: Tensor,
    pub exponent: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub base_requires_grad: bool,
    pub exp_requires_grad: bool,
    pub broadcast: PowBroadcast,
}
impl GradientFunction for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        match self.output.dtype() {
            DataType::Float32 => {
                let base_slice = self.base.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    // d/db (b^e) = e * b^(e-1)
                    let values: Vec<f32> = match self.broadcast {
                        PowBroadcast::None => ternary_map(
                            exp_slice,
                            base_slice,
                            grad_out,
                            |e: f32, b: f32, g: f32| e * b.powf(e - 1.0) * g,
                        ),
                        PowBroadcast::BaseScalar => {
                            // The scalar base receives the sum over every output.
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            vec![accum]
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            binary_map(base_slice, grad_out, move |b: f32, g: f32| {
                                exp_val * b.powf(exp_val - 1.0) * g
                            })
                        }
                    };
                    let grad_data =
                        TensorData::from_vec::<f32>(values, self.base.dtype(), self.base.device());

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[0], grad_tensor)?;
                }

                if self.exp_requires_grad {
                    // d/de (b^e) = b^e * ln(b)
                    let values: Vec<f32> = match self.broadcast {
                        PowBroadcast::None => ternary_map(
                            out_slice,
                            base_slice,
                            grad_out,
                            |o: f32, b: f32, g: f32| o * b.ln() * g,
                        ),
                        PowBroadcast::BaseScalar => {
                            let log_base = base_slice[0].ln();
                            binary_map(out_slice, grad_out, move |o: f32, g: f32| o * log_base * g)
                        }
                        PowBroadcast::ExponentScalar => {
                            // The scalar exponent receives the sum over every output.
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            vec![accum]
                        }
                    };
                    let grad_data = TensorData::from_vec::<f32>(
                        values,
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[1], grad_tensor)?;
                }
            }
            DataType::Float64 => {
                let base_slice = self.base.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    // d/db (b^e) = e * b^(e-1)
                    let values: Vec<f64> = match self.broadcast {
                        PowBroadcast::None => ternary_map(
                            exp_slice,
                            base_slice,
                            grad_out,
                            |e: f64, b: f64, g: f64| e * b.powf(e - 1.0) * g,
                        ),
                        PowBroadcast::BaseScalar => {
                            // The scalar base receives the sum over every output.
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            vec![accum]
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            binary_map(base_slice, grad_out, move |b: f64, g: f64| {
                                exp_val * b.powf(exp_val - 1.0) * g
                            })
                        }
                    };
                    let grad_data =
                        TensorData::from_vec::<f64>(values, self.base.dtype(), self.base.device());

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[0], grad_tensor)?;
                }

                if self.exp_requires_grad {
                    // d/de (b^e) = b^e * ln(b)
                    let values: Vec<f64> = match self.broadcast {
                        PowBroadcast::None => ternary_map(
                            out_slice,
                            base_slice,
                            grad_out,
                            |o: f64, b: f64, g: f64| o * b.ln() * g,
                        ),
                        PowBroadcast::BaseScalar => {
                            let log_base = base_slice[0].ln();
                            binary_map(out_slice, grad_out, move |o: f64, g: f64| o * log_base * g)
                        }
                        PowBroadcast::ExponentScalar => {
                            // The scalar exponent receives the sum over every output.
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            vec![accum]
                        }
                    };
                    let grad_data = TensorData::from_vec::<f64>(
                        values,
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, self.input_ids[1], grad_tensor)?;
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Power backward only supported for floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
