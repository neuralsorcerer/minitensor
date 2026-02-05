// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let exp_ptr = exp_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f32;
                                    let exp_ptr = exp_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) = *exp_ptr.add(i)
                                        * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[0], grad_tensor);
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = exp_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                                }
                            } else {
                                let out_ptr = out_slice.as_ptr() as usize;
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let out_ptr = out_ptr as *const f32;
                                    let base_ptr = base_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) =
                                        *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[1], grad_tensor);
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
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let exp_ptr = exp_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f64;
                                    let exp_ptr = exp_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) = *exp_ptr.add(i)
                                        * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[0], grad_tensor);
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = exp_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                                }
                            } else {
                                let out_ptr = out_slice.as_ptr() as usize;
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let out_ptr = out_ptr as *const f64;
                                    let base_ptr = base_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) =
                                        *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                                });
                            }
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                    }

                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[1], grad_tensor);
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

/// Gradient function for Hardshrink
pub struct HardshrinkBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for HardshrinkBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "hardshrink backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for ReLU
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for LeakyReLU
pub struct LeakyReluBackward {
    pub input_id: TensorId,
    pub negative_slope: f64,
    pub mask: Vec<bool>,
}

impl GradientFunction for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope as f32;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LeakyReLU backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for softmax
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Allocate gradient buffer
        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                softmax_backward_f32(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                softmax_backward_f64(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softmax backward only supported for floating point tensors",
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

/// Gradient function for log-softmax
pub struct LogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f32(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f64(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax backward only supported for floating point tensors",
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

/// Gradient function for masked log-softmax
pub struct MaskedLogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub mask: Tensor,
    pub dim: usize,
}
