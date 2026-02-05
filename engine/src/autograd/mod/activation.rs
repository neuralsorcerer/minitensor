// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn repeat_interleave_backward_impl(
    grad_output: &Tensor,
    input_shape: &[usize],
    repeats: &[usize],
    dim: usize,
) -> Result<Tensor> {
    if dim >= input_shape.len() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            input_shape.len(),
        ));
    }

    let dim_size = input_shape[dim];
    if repeats.len() != dim_size {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave backward: repeats must match input dimension size".to_string(),
        ));
    }

    let grad_shape_vec = input_shape.to_vec();
    let grad_shape = Shape::new(grad_shape_vec.clone());
    let numel = grad_shape.numel();
    let dtype = grad_output.dtype();
    let device = grad_output.device();
    let total_repeats: usize = repeats.iter().sum();

    let inner: usize = if dim + 1 >= input_shape.len() {
        1
    } else {
        input_shape[dim + 1..].iter().product()
    };
    let outer: usize = if dim == 0 {
        1
    } else {
        input_shape[..dim].iter().product()
    };

    if numel == 0 || total_repeats == 0 || inner == 0 || outer == 0 {
        return Ok(Tensor::zeros(
            Shape::new(grad_shape_vec),
            dtype,
            device,
            false,
        ));
    }

    let output_dims = grad_output.shape().dims();
    if output_dims.len() != input_shape.len() || output_dims[dim] != total_repeats {
        return Err(MinitensorError::shape_mismatch(
            input_shape.to_vec(),
            output_dims.to_vec(),
        ));
    }

    macro_rules! repeat_interleave_backward_impl_inner {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = grad_output.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave backward: gradient tensor must be contiguous".to_string(),
                )
            })?;
            let mut dst = vec![<$ty>::default(); numel];
            let chunk = total_repeats * inner;
            dst.par_chunks_mut(dim_size * inner)
                .enumerate()
                .for_each(|(outer_idx, dst_chunk)| {
                    let mut src_offset = outer_idx * chunk;
                    for (i, &rep) in repeats.iter().enumerate() {
                        if rep == 0 {
                            continue;
                        }
                        let dst_start = i * inner;
                        let dst_slice = &mut dst_chunk[dst_start..dst_start + inner];
                        for _ in 0..rep {
                            let src_slice = &src[src_offset..src_offset + inner];
                            dst_slice.iter_mut().zip(src_slice.iter()).for_each(
                                |(dst_val, &src_val)| {
                                    *dst_val += src_val;
                                },
                            );
                            src_offset += inner;
                        }
                    }
                });
            TensorData::$from_vec(dst, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => {
            repeat_interleave_backward_impl_inner!(f32, as_f32_slice, from_vec_f32)
        }
        DataType::Float64 => {
            repeat_interleave_backward_impl_inner!(f64, as_f64_slice, from_vec_f64)
        }
        DataType::Int32 => repeat_interleave_backward_impl_inner!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_interleave_backward_impl_inner!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => {
            return Ok(Tensor::zeros(grad_shape, dtype, device, false));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        grad_shape,
        dtype,
        device,
        false,
    ))
}

/// Gradient function for expand operation which reduces broadcasted gradients
pub struct ExpandBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = Shape::new(self.input_shape.clone());
        let grad_input = reduce_gradient_for_broadcasting(grad_output, &shape)?;
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for MSE loss
pub struct MSELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub diff: Tensor,
}

impl GradientFunction for MSELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Base gradient: 2 * (predictions - targets)
        let two = create_scalar_tensor(2.0, self.diff.dtype(), self.diff.device())?;
        let mut base_grad = arithmetic::mul(&self.diff, &two)?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let n = self.diff.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        // Multiply by upstream gradient
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for MAE loss
pub struct MAELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub sign: Tensor,
}

impl GradientFunction for MAELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let mut base_grad = self.sign.clone();
        match self.reduction.as_str() {
            "mean" => {
                let n = self.sign.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Huber loss
pub struct HuberLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub delta: f64,
    pub reduction: String,
    pub diff: Tensor,
}

impl GradientFunction for HuberLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let numel = self.diff.numel();
        let dtype = self.diff.dtype();
        let device = self.diff.device();
        let mut grad_data = TensorData::zeros_on_device(numel, dtype, device);

        match dtype {
            DataType::Float32 => {
                let diff_slice = self.diff.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from diff")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f32 slice from grad")
                })?;
                let delta = self.delta as f32;
                if numel < PAR_THRESHOLD {
                    for i in 0..numel {
                        let d = diff_slice[i];
                        grad_slice[i] = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    }
                } else {
                    let diff_ptr = diff_slice.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..numel).into_par_iter().for_each(|i| unsafe {
                        let diff_ptr = diff_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let d = *diff_ptr.add(i);
                        *grad_ptr.add(i) = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    });
                }
            }
            DataType::Float64 => {
                let diff_slice = self.diff.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from diff")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f64 slice from grad")
                })?;
                if numel < PAR_THRESHOLD {
                    for i in 0..numel {
                        let d = diff_slice[i];
                        grad_slice[i] = if d.abs() <= self.delta {
                            d
                        } else {
                            self.delta * d.signum()
                        };
                    }
                } else {
                    let diff_ptr = diff_slice.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    let delta = self.delta;
                    (0..numel).into_par_iter().for_each(|i| unsafe {
                        let diff_ptr = diff_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let d = *diff_ptr.add(i);
                        *grad_ptr.add(i) = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Huber loss only supports floating point tensors",
                ));
            }
        }

        let mut base_grad = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.predictions_shape.clone()),
            dtype,
            device,
            false,
        );

        if self.reduction == "mean" {
            let scale = create_scalar_tensor(1.0 / numel as f64, dtype, device)?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Cross Entropy loss
pub struct CrossEntropyLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for CrossEntropyLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Compute base gradient: softmax(predictions) - targets
        let mut base_grad =
            arithmetic::sub(&self.softmax_predictions.detach(), &self.targets.detach())?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let batch = self.targets_shape[0] as f64;
                let mut scalar_data =
                    TensorData::zeros_on_device(1, base_grad.dtype(), base_grad.device());
                match base_grad.dtype() {
                    DataType::Float32 => {
                        let slice = scalar_data.as_f32_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f32 slice from scalar",
                            )
                        })?;
                        slice[0] = (1.0 / batch) as f32;
                    }
                    DataType::Float64 => {
                        let slice = scalar_data.as_f64_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f64 slice from scalar",
                            )
                        })?;
                        slice[0] = 1.0 / batch;
                    }
                    _ => {
                        return Err(MinitensorError::invalid_operation(
                            "CrossEntropy backward only supports floating point tensors",
                        ));
                    }
                }
                let scalar_tensor = Tensor::new(
                    Arc::new(scalar_data),
                    Shape::new(vec![1]),
                    base_grad.dtype(),
                    base_grad.device(),
                    false,
                );
                base_grad = arithmetic::mul(&base_grad, &scalar_tensor)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )));
            }
        }

        // Multiply by upstream gradient (handles broadcasting)
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;

        // Targets typically have no gradient
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Binary Cross Entropy loss
pub struct BCELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for BCELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // BCE gradient: (predictions - targets) / (predictions * (1 - predictions))
        let one = Tensor::ones(
            Shape::new(self.predictions_shape.clone()),
            self.predictions.dtype(),
            self.predictions.device(),
            false,
        );
        let one_minus_pred = arithmetic::sub(&one, &self.predictions)?;
        let numerator = arithmetic::sub(&self.predictions, &self.targets)?;
        let denom = arithmetic::mul(&self.predictions, &one_minus_pred)?;
        let mut base_grad = arithmetic::div(&numerator, &denom)?;

        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for KL Divergence loss
pub struct KLDivLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for KLDivLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Gradient w.r.t predictions: -(targets / predictions)
        let mut pred_grad = arithmetic::div(&self.targets, &self.predictions)?;
        pred_grad = arithmetic::neg(&pred_grad)?;
        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, pred_grad.dtype(), pred_grad.device())?;
            pred_grad = arithmetic::mul(&pred_grad, &scale)?;
        }
        let pred_grad = arithmetic::mul(&pred_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        // Gradient w.r.t targets: log(targets) - log(predictions) + 1
        let log_targets = activation::log(&self.targets)?;
        let log_preds = activation::log(&self.predictions)?;
        let diff = arithmetic::sub(&log_targets, &log_preds)?;
        let one = Tensor::ones(
            self.targets.shape().clone(),
            self.targets.dtype(),
            self.targets.device(),
            false,
        );
        let mut target_grad = arithmetic::add(&diff, &one)?;
        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, target_grad.dtype(), target_grad.device())?;
            target_grad = arithmetic::mul(&target_grad, &scale)?;
        }
        let target_grad = arithmetic::mul(&target_grad, grad_output)?;
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Focal loss
pub struct FocalLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub alpha: f64,
    pub gamma: f64,
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for FocalLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Compute base gradient similar to cross entropy
        let p = self.softmax_predictions.detach();
        let t = self.targets.detach();
        let mut base_grad = arithmetic::sub(&p, &t)?;

        // Compute focal weight: alpha * (1 - p)^gamma
        let one = Tensor::ones(p.shape().clone(), p.dtype(), p.device(), false);
        let one_minus_p = arithmetic::sub(&one, &p)?;
        let mut weight = tensor_power(&one_minus_p, self.gamma)?;
        let alpha_tensor = create_scalar_tensor(self.alpha, p.dtype(), p.device())?;
        weight = arithmetic::mul(&weight, &alpha_tensor)?;

        base_grad = arithmetic::mul(&base_grad, &weight)?;

        if self.reduction == "mean" {
            let batch = self.targets_shape[0] as f64;
            let scale = create_scalar_tensor(1.0 / batch, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Create a scalar tensor with the given value
fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from scalar")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from scalar")
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensors only supported for floating point types",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
}

/// Raise each tensor element to the given power
fn tensor_power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;
            let exp = exponent as f32;
            let len = input.len();
            debug_assert_eq!(len, output.len());
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    output[i] = input[i].powf(exp);
                }
            } else {
                let in_ptr = input.as_ptr() as usize;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let in_ptr = in_ptr as *const f32;
                    let out_ptr = out_ptr as *mut f32;
                    *out_ptr.add(i) = (*in_ptr.add(i)).powf(exp);
                });
            }
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;
            let len = input.len();
            debug_assert_eq!(len, output.len());
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    output[i] = input[i].powf(exponent);
                }
            } else {
                let in_ptr = input.as_ptr() as usize;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let in_ptr = in_ptr as *const f64;
                    let out_ptr = out_ptr as *mut f64;
                    *out_ptr.add(i) = (*in_ptr.add(i)).powf(exponent);
                });
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}
