// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

impl GradientFunction for MaskedLogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        let output_dims = self.output.shape().dims();
        let mask_dims = self.mask.shape().dims();
        let same_shape = output_dims == mask_dims;
        let output_strides = if same_shape {
            None
        } else {
            Some(Strides::from_shape(self.output.shape()))
        };
        let mask_strides = if same_shape {
            None
        } else {
            Some(Strides::from_shape(self.mask.shape()))
        };

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from masked log_softmax output",
                    )
                })?;
                let mask_data = self.mask.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from mask tensor")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                masked_log_softmax_backward_f32(
                    go,
                    log_y,
                    mask_data,
                    grad_slice,
                    output_dims,
                    self.dim,
                    mask_dims,
                    output_strides.as_ref().map(Strides::as_slice),
                    mask_strides.as_ref().map(Strides::as_slice),
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from masked log_softmax output",
                    )
                })?;
                let mask_data = self.mask.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from mask tensor")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                masked_log_softmax_backward_f64(
                    go,
                    log_y,
                    mask_data,
                    grad_slice,
                    output_dims,
                    self.dim,
                    mask_dims,
                    output_strides.as_ref().map(Strides::as_slice),
                    mask_strides.as_ref().map(Strides::as_slice),
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Masked log_softmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for layer normalization
pub struct LayerNormBackward {
    pub input_ids: SmallVec<[TensorId; 3]>,
    pub input_id: TensorId,
    pub weight_id: Option<TensorId>,
    pub bias_id: Option<TensorId>,
    pub normalized: Tensor,
    pub inv_std: Tensor,
    pub weight_broadcast: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub axis_start: usize,
    pub element_count: usize,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
}

impl GradientFunction for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        let grad_output_detached = grad_output.detach();
        let normalized = self.normalized.detach();

        if self.element_count == 0 {
            if self.input_requires_grad {
                let zero = Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                );
                gradients.insert(self.input_id, zero);
            }
            if self.weight_requires_grad {
                if let Some(weight_id) = self.weight_id {
                    let zero = Tensor::zeros(
                        Shape::new(self.normalized_shape.clone()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    gradients.insert(weight_id, zero);
                }
            }
            if self.bias_requires_grad {
                if let Some(bias_id) = self.bias_id {
                    let zero = Tensor::zeros(
                        Shape::new(self.normalized_shape.clone()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    gradients.insert(bias_id, zero);
                }
            }

            return Ok(gradients);
        }

        if self.input_requires_grad {
            let mut grad_output_hat = if let Some(weight) = &self.weight_broadcast {
                arithmetic::mul(&grad_output_detached, weight)?
            } else {
                grad_output_detached.clone()
            };

            let axes: Vec<isize> = (self.axis_start..grad_output_hat.ndim())
                .map(|d| d as isize)
                .collect();
            let sum_grad = reduction::sum(&grad_output_hat, Some(axes.clone()), true)?;
            let grad_norm_mul = arithmetic::mul(&grad_output_hat, &normalized)?;
            let sum_grad_norm = reduction::sum(&grad_norm_mul, Some(axes), true)?;

            let count = self.element_count as f64;
            let m_tensor = create_scalar_tensor(count, grad_output.dtype(), grad_output.device())?;
            let inv_m_tensor =
                create_scalar_tensor(1.0 / count, grad_output.dtype(), grad_output.device())?;
            grad_output_hat = arithmetic::mul(&grad_output_hat, &m_tensor)?;
            let tmp = arithmetic::sub(&grad_output_hat, &sum_grad)?;
            let norm_term = arithmetic::mul(&normalized, &sum_grad_norm)?;
            let numerator = arithmetic::sub(&tmp, &norm_term)?;
            let grad_input = arithmetic::mul(&numerator, &self.inv_std)?;
            let grad_input = arithmetic::mul(&grad_input, &inv_m_tensor)?;
            gradients.insert(self.input_id, grad_input);
        }

        if self.weight_requires_grad {
            if let Some(weight_id) = self.weight_id {
                let mut grad_weight = arithmetic::mul(&grad_output_detached, &normalized)?;
                if self.axis_start > 0 {
                    let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                    grad_weight = reduction::sum(&grad_weight, Some(axes), false)?;
                }
                if grad_weight.shape().dims() != self.normalized_shape.as_slice() {
                    grad_weight = grad_weight.view(Shape::new(self.normalized_shape.clone()))?;
                }
                gradients.insert(weight_id, grad_weight);
            }
        }

        if self.bias_requires_grad {
            if let Some(bias_id) = self.bias_id {
                let mut grad_bias = grad_output_detached.clone();
                if self.axis_start > 0 {
                    let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                    grad_bias = reduction::sum(&grad_bias, Some(axes), false)?;
                }
                if grad_bias.shape().dims() != self.normalized_shape.as_slice() {
                    grad_bias = grad_bias.view(Shape::new(self.normalized_shape.clone()))?;
                }
                gradients.insert(bias_id, grad_bias);
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

fn softmax_backward_f32(
    grad_output: &[f32],
    y: &[f32],
    grad_input: &mut [f32],
    dims: &[usize],
    dim: usize,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, y_block), out_block) in grad_output
            .chunks(group)
            .zip(y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut dot = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    dot += go_block[idx] * y_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, y_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut dot = 0.0f32;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        dot += go_block[idx] * y_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                    }
                }
            });
    }
}

fn softmax_backward_f64(
    grad_output: &[f64],
    y: &[f64],
    grad_input: &mut [f64],
    dims: &[usize],
    dim: usize,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, y_block), out_block) in grad_output
            .chunks(group)
            .zip(y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut dot = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    dot += go_block[idx] * y_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, y_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut dot = 0.0f64;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        dot += go_block[idx] * y_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                    }
                }
            });
    }
}

fn log_softmax_backward_f32(
    grad_output: &[f32],
    log_y: &[f32],
    grad_input: &mut [f32],
    dims: &[usize],
    dim: usize,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, log_block), out_block) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += go_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let prob = log_block[idx].exp();
                    out_block[idx] = go_block[idx] - prob * sum;
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, log_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f32;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        sum += go_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            });
    }
}

fn log_softmax_backward_f64(
    grad_output: &[f64],
    log_y: &[f64],
    grad_input: &mut [f64],
    dims: &[usize],
    dim: usize,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, log_block), out_block) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += go_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let prob = log_block[idx].exp();
                    out_block[idx] = go_block[idx] - prob * sum;
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, log_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f64;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        sum += go_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            });
    }
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

fn masked_log_softmax_backward_f32(
    grad_output: &[f32],
    log_y: &[f32],
    mask: &[bool],
    grad_input: &mut [f32],
    dims: &[usize],
    dim: usize,
    mask_dims: &[usize],
    output_strides: Option<&[usize]>,
    mask_strides: Option<&[usize]>,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = output_strides.is_none();

    if grad_output.len() < PAR_THRESHOLD {
        for (((go_block, log_block), out_block), block_idx) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
            .zip(0..)
        {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.unwrap(),
                            mask_dims,
                            mask_strides.unwrap(),
                        );
                        mask[mask_index]
                    };
                    if !masked {
                        sum += go_block[idx];
                    }
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.unwrap(),
                            mask_dims,
                            mask_strides.unwrap(),
                        );
                        mask[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .enumerate()
            .for_each(|(block_idx, ((go_block, log_block), out_block))| {
                let block_offset = block_idx * group;
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f32;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        let masked = if same_shape {
                            mask[linear_idx]
                        } else {
                            let mask_index = broadcast_mask_index(
                                linear_idx,
                                dims,
                                output_strides.unwrap(),
                                mask_dims,
                                mask_strides.unwrap(),
                            );
                            mask[mask_index]
                        };
                        if !masked {
                            sum += go_block[idx];
                        }
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        let masked = if same_shape {
                            mask[linear_idx]
                        } else {
                            let mask_index = broadcast_mask_index(
                                linear_idx,
                                dims,
                                output_strides.unwrap(),
                                mask_dims,
                                mask_strides.unwrap(),
                            );
                            mask[mask_index]
                        };
                        if masked {
                            out_block[idx] = 0.0;
                        } else {
                            let prob = log_block[idx].exp();
                            out_block[idx] = go_block[idx] - prob * sum;
                        }
                    }
                }
            });
    }
}

fn masked_log_softmax_backward_f64(
    grad_output: &[f64],
    log_y: &[f64],
    mask: &[bool],
    grad_input: &mut [f64],
    dims: &[usize],
    dim: usize,
    mask_dims: &[usize],
    output_strides: Option<&[usize]>,
    mask_strides: Option<&[usize]>,
) {
    if dims.is_empty() {
        if let Some(first) = grad_input.first_mut() {
            *first = 0.0;
        }
        return;
    }

    let dim_size = dims[dim];
    if dim_size == 0 {
        return;
    }

    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    let same_shape = output_strides.is_none();

    if grad_output.len() < PAR_THRESHOLD {
        for (((go_block, log_block), out_block), block_idx) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
            .zip(0..)
        {
            let block_offset = block_idx * group;
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.unwrap(),
                            mask_dims,
                            mask_strides.unwrap(),
                        );
                        mask[mask_index]
                    };
                    if !masked {
                        sum += go_block[idx];
                    }
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let linear_idx = block_offset + idx;
                    let masked = if same_shape {
                        mask[linear_idx]
                    } else {
                        let mask_index = broadcast_mask_index(
                            linear_idx,
                            dims,
                            output_strides.unwrap(),
                            mask_dims,
                            mask_strides.unwrap(),
                        );
                        mask[mask_index]
                    };
                    if masked {
                        out_block[idx] = 0.0;
                    } else {
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .enumerate()
            .for_each(|(block_idx, ((go_block, log_block), out_block))| {
                let block_offset = block_idx * group;
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f64;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        let masked = if same_shape {
                            mask[linear_idx]
                        } else {
                            let mask_index = broadcast_mask_index(
                                linear_idx,
                                dims,
                                output_strides.unwrap(),
                                mask_dims,
                                mask_strides.unwrap(),
                            );
                            mask[mask_index]
                        };
                        if !masked {
                            sum += go_block[idx];
                        }
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let linear_idx = block_offset + idx;
                        let masked = if same_shape {
                            mask[linear_idx]
                        } else {
                            let mask_index = broadcast_mask_index(
                                linear_idx,
                                dims,
                                output_strides.unwrap(),
                                mask_dims,
                                mask_strides.unwrap(),
                            );
                            mask[mask_index]
                        };
                        if masked {
                            out_block[idx] = 0.0;
                        } else {
                            let prob = log_block[idx].exp();
                            out_block[idx] = go_block[idx] - prob * sum;
                        }
                    }
                }
            });
    }
}

/// Gradient function for reshape operation
pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Reshape gradient: reshape back to original shape
        let original_shape = Shape::new(self.input_shape.clone());
        let grad_input = crate::operations::shape_ops::reshape(grad_output, original_shape)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for repeat_interleave operation
pub struct RepeatInterleaveBackward {
    pub input_shape: Vec<usize>,
    pub repeats: Vec<usize>,
    pub input_id: TensorId,
    pub dim: usize,
}

impl GradientFunction for RepeatInterleaveBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input = repeat_interleave_backward_impl(
            grad_output,
            &self.input_shape,
            &self.repeats,
            self.dim,
        )?;

        let mut gradients = FxHashMap::default();
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
