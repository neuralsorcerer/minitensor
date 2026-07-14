// Copyright (c) Soumyadip Sarkar.
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

        accumulate_grad(&mut gradients, self.input_id, grad_input)?;

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
                accumulate_grad(&mut gradients, self.input_id, zero)?;
            }
            if self.weight_requires_grad {
                if let Some(weight_id) = self.weight_id {
                    let zero = Tensor::zeros(
                        Shape::new(self.normalized_shape.clone()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    accumulate_grad(&mut gradients, weight_id, zero)?;
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
                    accumulate_grad(&mut gradients, bias_id, zero)?;
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
            accumulate_grad(&mut gradients, self.input_id, grad_input)?;
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
                accumulate_grad(&mut gradients, weight_id, grad_weight)?;
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
                accumulate_grad(&mut gradients, bias_id, grad_bias)?;
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
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;

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
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `min`/`max` reductions (global with `dim == None`, or
/// along a single `dim`).
///
/// The gradient flows to every input element equal to the reduced extremum,
/// split equally among ties so the contributions sum to the upstream gradient. The
/// extremum, its selection mask and the tie count are recomputed from the stored
/// (detached) input, so nothing beyond the input needs to be retained.
pub struct MinMaxBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub is_max: bool,
    pub nan_aware: bool,
}

/// Route `grad_output` to every input element equal to the selected reduction
/// value (`reduced`, recomputed with keepdim so it broadcasts), splitting equally
/// among ties. Shared by min/max and median value reductions.
fn distribute_selection_grad(
    input: &Tensor,
    reduced: &Tensor,
    grad_output: &Tensor,
    dim: Option<usize>,
    keepdim: bool,
) -> Result<Tensor> {
    let input_shape = input.shape().dims().to_vec();
    let mask = crate::operations::comparison::eq(input, reduced)?;
    let mask_f = mask.astype(input.dtype())?;

    let sum_dims = dim.map(|d| vec![d as isize]);
    let count = reduction::sum(&mask_f, sum_dims, true)?;

    let dims_vec = dim.map(|d| vec![d]);
    let grad_kd = expand_reduction_grad(grad_output, &input_shape, &dims_vec, keepdim)?;
    let scaled = arithmetic::div(&grad_kd, &count)?;
    arithmetic::mul(&mask_f, &scaled)
}

impl GradientFunction for MinMaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;

        if input.numel() == 0 {
            let zero = Tensor::zeros(input.shape().clone(), input.dtype(), input.device(), false);
            accumulate_grad(&mut gradients, self.input_id, zero)?;
            return Ok(gradients);
        }

        let dim_isize = self.dim.map(|d| d as isize);
        // Recompute the extremum with keepdim so it broadcasts against the input.
        // NaN-aware reductions must recompute with the matching op, otherwise the
        // propagated NaN would fail the equality mask and zero every gradient.
        let reduced = match (self.is_max, self.nan_aware) {
            (true, false) => reduction::max(input, dim_isize, true)?,
            (false, false) => reduction::min(input, dim_isize, true)?,
            (true, true) => reduction::nanmax(input, dim_isize, true)?,
            (false, true) => reduction::nanmin(input, dim_isize, true)?,
        };
        let grad_input =
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.keepdim)?;
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for `median`/`nanmedian` value reductions. The median is one
/// of the input elements, so the gradient flows to every element equal to it,
/// split over ties (a valid subgradient, matching the min/max convention).
pub struct MedianBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub keepdim: bool,
    pub nan_aware: bool,
}

impl GradientFunction for MedianBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;

        if input.numel() == 0 {
            let zero = Tensor::zeros(input.shape().clone(), input.dtype(), input.device(), false);
            accumulate_grad(&mut gradients, self.input_id, zero)?;
            return Ok(gradients);
        }

        let dim_isize = self.dim.map(|d| d as isize);
        let reduced = if self.nan_aware {
            reduction::nanmedian(input, dim_isize, true)?
        } else {
            reduction::median(input, dim_isize, true)?.0
        };
        let grad_input =
            distribute_selection_grad(input, &reduced, grad_output, self.dim, self.keepdim)?;
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for the `quantile` reduction (global with `dim == None`, or
/// along a single `dim`).
///
/// A quantile is a fixed linear combination of the two order statistics that
/// bracket the requested position, so the gradient routes back to the two
/// original elements occupying those sorted ranks with the interpolation weights
/// (`Lower`/`Higher`/`Nearest` collapse to a single element; `Midpoint` splits
/// evenly). Groups containing NaN produced NaN and receive no gradient.
pub struct QuantileBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub dim: Option<usize>,
    pub q: f64,
    pub interpolation: crate::operations::reduction::QuantileInterpolation,
    pub nan_aware: bool,
}

/// Sorted-rank indices and their gradient weights for a group of length `len`.
fn quantile_grad_coeffs(
    len: usize,
    q: f64,
    interp: crate::operations::reduction::QuantileInterpolation,
) -> (usize, usize, f64, f64) {
    use crate::operations::reduction::QuantileInterpolation as Qi;
    if len <= 1 {
        return (0, 0, 1.0, 0.0);
    }
    let pos = q * (len - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let weight = (pos - lower as f64).clamp(0.0, 1.0);
    match interp {
        Qi::Linear => (lower, upper, 1.0 - weight, weight),
        Qi::Lower => (lower, upper, 1.0, 0.0),
        Qi::Higher => (lower, upper, 0.0, 1.0),
        Qi::Midpoint => (lower, upper, 0.5, 0.5),
        Qi::Nearest => {
            // Ties at weight == 0.5 round to the even index.
            let nearest = if weight < 0.5 {
                lower
            } else if weight > 0.5 {
                upper
            } else {
                lower + (lower & 1)
            };
            if nearest == lower {
                (lower, upper, 1.0, 0.0)
            } else {
                (lower, upper, 0.0, 1.0)
            }
        }
    }
}

impl GradientFunction for QuantileBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let input = &self.input;
        let numel = input.numel();
        let mut grad_data = TensorData::zeros_on_device(numel, input.dtype(), input.device());

        if numel != 0 {
            // Treat the global reduction as one group over the flattened tensor.
            let dims = input.shape().dims();
            let (outer, inner, dim_size) = match self.dim {
                None => (1usize, 1usize, numel),
                Some(d) => {
                    let outer: usize = dims[..d].iter().product();
                    let inner: usize = dims[d + 1..].iter().product();
                    (outer, inner, dims[d])
                }
            };
            let outer_stride = dim_size * inner;

            macro_rules! scatter {
                ($slice:ident, $mut_slice:ident, $ty:ty) => {{
                    let x = input.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to read input for quantile backward")
                    })?;
                    let go = grad_output.data().$slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to read grad_output for quantile backward",
                        )
                    })?;
                    let gi = grad_data.$mut_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to write grad for quantile backward")
                    })?;
                    let mut buffer: Vec<(usize, $ty)> = Vec::with_capacity(dim_size);
                    for o in 0..outer {
                        for r in 0..inner {
                            buffer.clear();
                            let mut skip_group = false;
                            for d in 0..dim_size {
                                let v = x[o * outer_stride + d * inner + r];
                                if v.is_nan() {
                                    if self.nan_aware {
                                        // nanquantile ignores NaN entries.
                                        continue;
                                    }
                                    // quantile propagates NaN, so the whole group's
                                    // output is NaN and gets no gradient.
                                    skip_group = true;
                                    break;
                                }
                                buffer.push((d, v));
                            }
                            if skip_group || buffer.is_empty() {
                                continue;
                            }
                            let (lo, up, c_lo, c_up) =
                                quantile_grad_coeffs(buffer.len(), self.q, self.interpolation);
                            // Only the elements at sorted ranks `lo` and `up`
                            // (adjacent) are needed, so select them in O(n) rather
                            // than fully sorting. NaN is already filtered, so the
                            // comparator never sees an incomparable value.
                            let cmp = |a: &(usize, $ty), b: &(usize, $ty)| {
                                a.1.partial_cmp(&b.1).unwrap()
                            };
                            buffer.select_nth_unstable_by(up, cmp);
                            let d_up = buffer[up].0;
                            let d_lo = if lo == up {
                                d_up
                            } else {
                                // `lo == up - 1`: the lo-th order statistic is the
                                // largest of the elements left of `up`.
                                buffer[..up].select_nth_unstable_by(lo, cmp);
                                buffer[lo].0
                            };
                            let g = go[o * inner + r];
                            gi[o * outer_stride + d_lo * inner + r] += g * c_lo as $ty;
                            gi[o * outer_stride + d_up * inner + r] += g * c_up as $ty;
                        }
                    }
                }};
            }

            match input.dtype() {
                DataType::Float32 => scatter!(as_f32_slice, as_f32_slice_mut, f32),
                DataType::Float64 => scatter!(as_f64_slice, as_f64_slice_mut, f64),
                _ => {
                    return Err(MinitensorError::invalid_operation(
                        "quantile backward only supported for floating point tensors",
                    ));
                }
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            input.shape().clone(),
            input.dtype(),
            input.device(),
            false,
        );
        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}
