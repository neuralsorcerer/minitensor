// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

impl Tensor {
    /// Solve a linear system `AX = B` for `X` where `self` provides `A`.
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        use crate::operations::linalg::solve;
        solve(self, rhs)
    }

    /// Layer normalization
    #[inline(always)]
    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Self> {
        use crate::operations::normalization::layer_norm;
        layer_norm(self, normalized_shape, weight, bias, eps)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(&self) -> Result<Self> {
        use crate::operations::activation::abs;
        abs(self)
    }

    /// Element-wise sign (returns -1, 0, or 1 for each value).
    #[inline(always)]
    pub fn sign(&self) -> Result<Self> {
        use crate::operations::activation::sign;
        sign(self)
    }

    /// Clip tensor values to the provided range.
    #[inline(always)]
    pub fn clip(&self, min_val: Option<f64>, max_val: Option<f64>) -> Result<Self> {
        if let (Some(min), Some(max)) = (min_val, max_val) {
            if min > max {
                return Err(MinitensorError::invalid_argument(format!(
                    "clip minimum {min} cannot be greater than maximum {max}",
                )));
            }
        }

        use crate::operations::activation::clip;
        clip(self, min_val, max_val)
    }

    /// Alias for [`Tensor::clip`] following PyTorch's `clamp` naming.
    #[inline(always)]
    pub fn clamp(&self, min_val: Option<f64>, max_val: Option<f64>) -> Result<Self> {
        self.clip(min_val, max_val)
    }

    /// Clamp tensor values to be no smaller than `min_val`.
    #[inline(always)]
    pub fn clamp_min(&self, min_val: f64) -> Result<Self> {
        self.clip(Some(min_val), None)
    }

    /// Clamp tensor values to be no larger than `max_val`.
    #[inline(always)]
    pub fn clamp_max(&self, max_val: f64) -> Result<Self> {
        self.clip(None, Some(max_val))
    }

    /// Round tensor values to a specific number of decimal places.
    #[inline(always)]
    pub fn round(&self, decimals: i32) -> Result<Self> {
        use crate::operations::activation::round;
        round(self, decimals)
    }

    /// Floor tensor values element-wise.
    #[inline(always)]
    pub fn floor(&self) -> Result<Self> {
        use crate::operations::activation::floor;
        floor(self)
    }

    /// Ceil tensor values element-wise.
    #[inline(always)]
    pub fn ceil(&self) -> Result<Self> {
        use crate::operations::activation::ceil;
        ceil(self)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(&self) -> Result<Self> {
        use crate::operations::activation::sqrt;
        sqrt(self)
    }

    pub fn rsqrt(&self) -> Result<Self> {
        use crate::operations::activation::rsqrt;
        rsqrt(self)
    }

    /// Element-wise reciprocal (1/x).
    #[inline(always)]
    pub fn reciprocal(&self) -> Result<Self> {
        use crate::operations::activation::reciprocal;
        reciprocal(self)
    }

    /// Raise tensor elements to a scalar power
    #[inline(always)]
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        use crate::operations::activation::powf;
        powf(self, exponent)
    }

    /// Numerically stable logaddexp
    #[inline(always)]
    pub fn logaddexp(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::activation::logaddexp;
        logaddexp(self, other)
    }

    /// Element-wise power with another tensor
    pub fn pow(&self, exponent: &Tensor) -> Result<Self> {
        use crate::operations::activation::pow;
        pow(self, exponent)
    }

    /// Move tensor to device
    #[inline(always)]
    pub fn to(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }

        // For now, just clone the tensor with the new device
        // In a full implementation, we'd copy data between devices
        let mut new_tensor = self.clone();
        new_tensor.device = device;
        Ok(new_tensor)
    }

    /// Convert tensor to a different data type
    #[inline(always)]
    pub fn astype(&self, dtype: DataType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }

        let numel = self.numel();
        let mut new_data = TensorData::zeros_on_device(numel, dtype, self.device);

        // Helper macro to cast between slices using parallel iteration for large buffers
        macro_rules! cast {
            ($src:expr, $dst:expr, $conv:expr) => {{
                if numel >= 1024 {
                    $dst.par_iter_mut().zip($src.par_iter()).for_each(|(d, s)| {
                        *d = $conv(*s);
                    });
                } else {
                    for (d, &s) in $dst.iter_mut().zip($src.iter()) {
                        *d = $conv(s);
                    }
                }
            }};
        }

        match (self.dtype, dtype) {
            (DataType::Float32, DataType::Float64) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as f64);
            }
            (DataType::Float32, DataType::Int32) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as i32);
            }
            (DataType::Float32, DataType::Int64) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as i64);
            }
            (DataType::Float32, DataType::Bool) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v != 0.0);
            }
            (DataType::Float64, DataType::Float32) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as f32);
            }
            (DataType::Float64, DataType::Int32) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as i32);
            }
            (DataType::Float64, DataType::Int64) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as i64);
            }
            (DataType::Float64, DataType::Bool) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v != 0.0);
            }
            (DataType::Int32, DataType::Float32) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as f32);
            }
            (DataType::Int32, DataType::Float64) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as f64);
            }
            (DataType::Int32, DataType::Int64) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as i64);
            }
            (DataType::Int32, DataType::Bool) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v != 0);
            }
            (DataType::Int64, DataType::Float32) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as f32);
            }
            (DataType::Int64, DataType::Float64) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as f64);
            }
            (DataType::Int64, DataType::Int32) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as i32);
            }
            (DataType::Int64, DataType::Bool) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v != 0);
            }
            (DataType::Bool, DataType::Float32) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1.0 } else { 0.0 });
            }
            (DataType::Bool, DataType::Float64) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1.0 } else { 0.0 });
            }
            (DataType::Bool, DataType::Int32) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1 } else { 0 });
            }
            (DataType::Bool, DataType::Int64) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1 } else { 0 });
            }
            _ => unreachable!("Unhandled dtype conversion"),
        }

        Ok(Tensor::new(
            Arc::new(new_data),
            self.shape.clone(),
            dtype,
            self.device,
            self.requires_grad,
        ))
    }
}

impl Tensor {
    /// Copy data from ``source`` into this tensor in-place, preserving dtype and device.
    pub fn copy_(&mut self, source: &Tensor) -> Result<()> {
        if self.shape != *source.shape() {
            return Err(MinitensorError::invalid_argument(format!(
                "copy_ expected source with shape {:?}, but received {:?}",
                self.shape.dims(),
                source.shape().dims()
            )));
        }

        if !self.device.is_cpu() {
            return Err(MinitensorError::invalid_operation(
                "copy_ currently supports only CPU tensors".to_string(),
            ));
        }

        let mut prepared: Cow<'_, Tensor> = Cow::Borrowed(source);

        if prepared.dtype() != self.dtype {
            prepared = Cow::Owned(prepared.astype(self.dtype)?);
        }

        if prepared.device() != self.device {
            prepared = Cow::Owned(prepared.to(self.device)?);
        }

        if !prepared.is_contiguous() {
            prepared = Cow::Owned(prepared.contiguous()?);
        }

        if !self.is_contiguous() {
            return Err(MinitensorError::invalid_operation(
                "copy_ currently requires the destination tensor to be contiguous".to_string(),
            ));
        }

        let dtype = self.dtype;
        {
            let dst_data = self.data_mut();
            match dtype {
                DataType::Float32 => {
                    let dst = dst_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float32 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access float32 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Float64 => {
                    let dst = dst_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float64 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access float64 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Int32 => {
                    let dst = dst_data.as_i32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int32 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_i32_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access int32 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Int64 => {
                    let dst = dst_data.as_i64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int64 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_i64_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access int64 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Bool => {
                    let dst = dst_data.as_bool_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable bool slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_bool_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access bool source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
            }
        }

        self.refresh_autograd_metadata();
        if self.requires_grad {
            autograd::add_to_graph(self, None)?;
        }

        Ok(())
    }

    /// Fill the tensor in-place with ``value`` converted to the tensor dtype.
    pub fn fill_(&mut self, value: f64) -> Result<()> {
        if !self.device.is_cpu() {
            return Err(MinitensorError::invalid_operation(
                "fill_ currently supports only CPU tensors".to_string(),
            ));
        }

        if !self.is_contiguous() {
            return Err(MinitensorError::invalid_operation(
                "fill_ currently requires contiguous tensors".to_string(),
            ));
        }

        let dtype = self.dtype;
        {
            let data = self.data_mut();
            match dtype {
                DataType::Float32 => {
                    let slice = data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float32 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as f32);
                }
                DataType::Float64 => {
                    let slice = data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float64 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value);
                }
                DataType::Int32 => {
                    let slice = data.as_i32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int32 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as i32);
                }
                DataType::Int64 => {
                    let slice = data.as_i64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int64 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as i64);
                }
                DataType::Bool => {
                    let slice = data.as_bool_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable bool slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value != 0.0);
                }
            }
        }

        self.refresh_autograd_metadata();
        if self.requires_grad {
            autograd::add_to_graph(self, None)?;
        }

        Ok(())
    }
}

impl Tensor {
    /// Detach tensor from computation graph
    #[inline(always)]
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached.grad_fn = None;
        detached.grad = None;
        detached
    }

    /// Detach tensor from the computation graph in-place
    #[inline(always)]
    pub fn detach_inplace(&mut self) {
        self.requires_grad = false;
        self.refresh_autograd_metadata();
    }

    /// Check if tensors are approximately equal
    #[inline(always)]
    pub fn allclose(&self, other: &Tensor, rtol: f64, atol: f64) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        // Fast path: byte-for-byte equality check for contiguous CPU tensors
        if self.device.is_cpu()
            && other.device.is_cpu()
            && self.is_contiguous()
            && other.is_contiguous()
        {
            if let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes()) {
                if a == b {
                    return true;
                }
            }
        }

        let numel = self.numel();
        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| {
                                let diff = (a - b).abs();
                                diff <= atol as f32 + rtol as f32 * b.abs()
                            })
                    } else {
                        self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                            let diff = (a - b).abs();
                            diff <= atol as f32 + rtol as f32 * b.abs()
                        })
                    }
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| {
                                let diff = (a - b).abs();
                                diff <= atol + rtol * b.abs()
                            })
                    } else {
                        self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                            let diff = (a - b).abs();
                            diff <= atol + rtol * b.abs()
                        })
                    }
                } else {
                    false
                }
            }
            _ => self.array_equal(other),
        }
    }

    /// Check if tensors are exactly equal
    #[inline(always)]
    pub fn array_equal(&self, other: &Tensor) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        // Fast path for contiguous CPU tensors using raw bytes comparison
        if self.device.is_cpu()
            && other.device.is_cpu()
            && self.is_contiguous()
            && other.is_contiguous()
        {
            if let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes()) {
                return a == b;
            }
        }

        let numel = self.numel();
        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Int32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i32_slice(), other.data.as_i32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Int64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i64_slice(), other.data.as_i64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Bool => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_bool_slice(), other.data.as_bool_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
        }
    }
}
