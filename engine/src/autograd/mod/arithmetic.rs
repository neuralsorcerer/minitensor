// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn expand_reduction_grad(
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

    let numel = grad.numel();
    let mut new_data = TensorData::zeros_on_device(numel, grad.dtype(), grad.device());

    match grad.dtype() {
        DataType::Float32 => {
            let grad_src = grad
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let count_src = count
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let dst = new_data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            dst.par_iter_mut()
                .zip(grad_src.par_iter().zip(count_src.par_iter()))
                .for_each(|(out, (&g, &c))| {
                    *out = if c == 0.0 { 0.0 } else { g };
                });
        }
        DataType::Float64 => {
            let grad_src = grad
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let count_src = count
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let dst = new_data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            dst.par_iter_mut()
                .zip(grad_src.par_iter().zip(count_src.par_iter()))
                .for_each(|(out, (&g, &c))| {
                    *out = if c == 0.0 { 0.0 } else { g };
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmean backward only supports floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(new_data),
        grad.shape().clone(),
        grad.dtype(),
        grad.device(),
        false,
    ))
}

fn safe_count_for_nanmean(count: &Tensor) -> Result<Tensor> {
    let numel = count.numel();
    let mut new_data = TensorData::zeros_on_device(numel, count.dtype(), count.device());

    match count.dtype() {
        DataType::Float32 => {
            let src = count
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let dst = new_data
                .as_f32_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(out, &c)| {
                    *out = if c == 0.0 { 1.0 } else { c };
                });
        }
        DataType::Float64 => {
            let src = count
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let dst = new_data
                .as_f64_slice_mut()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(out, &c)| {
                    *out = if c == 0.0 { 1.0 } else { c };
                });
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nanmean backward only supports floating point tensors",
            ));
        }
    }

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

        let mut grad = grad_output.clone();
        if !self.keepdim {
            if let Some(dims) = &self.dims {
                let mut shape = grad.shape().dims().to_vec();
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                for &d in &sorted {
                    shape.insert(d, 1);
                }
                grad = shape_ops::reshape(&grad, Shape::new(shape))?;
            } else {
                grad = shape_ops::reshape(&grad, Shape::new(vec![1; self.input.ndim()]))?;
            }
        }

        let mut prod = self.result.clone();
        if !self.keepdim {
            if let Some(dims) = &self.dims {
                let mut shape = prod.shape().dims().to_vec();
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                for &d in &sorted {
                    shape.insert(d, 1);
                }
                prod = shape_ops::reshape(&prod, Shape::new(shape))?;
            } else {
                prod = shape_ops::reshape(&prod, Shape::new(vec![1; self.input.ndim()]))?;
            }
        }

        let div = arithmetic::div(&prod, &self.input)?;
        let grad_input = arithmetic::mul(&grad, &div)?;
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

/// Gradient function for tanh
pub struct TanhBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(tanh(x)) = (1 - tanh²(x)) * grad_output
        let y2 = arithmetic::mul(&self.output, &self.output)?;
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::sub(&ones, &y2)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for sigmoid
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x)) * grad_output
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let one_minus = arithmetic::sub(&ones, &self.output)?;
        let term = arithmetic::mul(&self.output, &one_minus)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for Softplus
pub struct SoftplusBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub beta: f64,
    pub threshold: f64,
}
